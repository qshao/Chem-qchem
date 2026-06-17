# qchem_gnn/adapt/methods/engine.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..backbone import build_graphs, embed_per_layer, load_backbone
from ..data import LabelNormalizer
from ..metrics import regression_metrics
from .base import LoadedAdapter, TrainResult


class EngineAdapterHead(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.exit_heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.output_dim = output_dim

    def forward(self, layer_tensors):
        h = torch.zeros_like(layer_tensors[0])
        preds = []
        for proj, alpha, head, x in zip(self.projections, self.alphas, self.exit_heads, layer_tensors):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            preds.append(head(h))
        return preds

    @torch.no_grad()
    def predict_ensemble(self, layer_tensors):
        preds = self.forward(layer_tensors)
        return torch.stack(preds, dim=0).mean(0).cpu().numpy()

    @torch.no_grad()
    def predict_early_exit(self, layer_tensors, tolerance=0.05, min_layers=1):
        B = layer_tensors[0].shape[0]
        h = torch.zeros_like(layer_tensors[0])
        accumulated = []
        active = torch.ones(B, dtype=torch.bool)
        final_pred = torch.zeros(B, self.output_dim)
        exit_layer = torch.full((B,), self.num_layers - 1, dtype=torch.long)
        for i, (proj, alpha, head, x) in enumerate(
            zip(self.projections, self.alphas, self.exit_heads, layer_tensors)
        ):
            gate = torch.sigmoid(alpha)
            h = proj(x) * gate + h * (1.0 - gate)
            pred = head(h)                     # [B, output_dim]
            accumulated.append(pred.mean(dim=1))
            if i >= min_layers and len(accumulated) >= 2:
                std = torch.stack(accumulated, dim=0).std(dim=0)
                exiting = active & (std < tolerance)
                if exiting.any():
                    exit_layer[exiting] = i
                    final_pred[exiting] = pred[exiting]
                    active[exiting] = False
        if active.any():
            final_pred[active] = self.forward(layer_tensors)[-1][active]
        return final_pred.cpu().numpy(), exit_layer.cpu().numpy()


class EngineMethod:
    name = "engine"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)
        h_dim = model_config["hidden_dim"]
        n_steps = model_config["num_message_passing_steps"]
        y = data.targets
        T = y.shape[1]

        layer_embs = embed_per_layer(data.graphs, backbone)
        norm = LabelNormalizer.fit(y[train_idx])

        scalers, tr, va, te = [], [], [], []
        for emb in layer_embs:
            mu = emb[train_idx].mean(0); sig = emb[train_idx].std(0).clip(1e-8)
            scalers.append((mu, sig))
            tr.append(torch.as_tensor((emb[train_idx] - mu) / sig, dtype=torch.float32))
            va.append(torch.as_tensor((emb[val_idx] - mu) / sig, dtype=torch.float32) if val_idx else None)
            te.append(torch.as_tensor((emb[test_idx] - mu) / sig, dtype=torch.float32) if test_idx else None)
        ytr = torch.as_tensor(norm.transform(y[train_idx]), dtype=torch.float32)

        adapter = EngineAdapterHead(h_dim, n_steps, output_dim=T)
        epochs = int(cfg.training.get("epochs", 400))
        lr = float(cfg.training.get("head_lr", cfg.training.get("lr", 3e-3)))
        opt = torch.optim.Adam(adapter.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        crit = nn.MSELoss()

        best_val = float("inf")
        best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
        for _ in range(epochs):
            adapter.train()
            opt.zero_grad()
            loss = sum(crit(p, ytr) for p in adapter(tr))
            loss.backward(); opt.step(); sched.step()
            if val_idx:
                adapter.eval()
                vp = norm.inverse(adapter.predict_ensemble(va))
                vmae = float(np.mean(np.abs(vp - y[val_idx])))
                if vmae < best_val:
                    best_val = vmae
                    best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

        adapter.load_state_dict(best_state); adapter.eval()
        test_metrics = {}
        if test_idx:
            tp = norm.inverse(adapter.predict_ensemble(te))
            test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "engine",
            "adapter_state": adapter.state_dict(),
            "hidden_dim": h_dim, "num_layers": n_steps, "output_dim": T,
            "layer_scalers": [(mu.tolist(), sig.tolist()) for mu, sig in scalers],
            "label_norm": norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_mae": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta,
                    "training_info": {**meta.get("training_info", {}),
                                      "test_metrics": result.test_metrics, **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="engine", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        model, _ = load_backbone(s["backbone_ckpt"])
        adapter = EngineAdapterHead(s["hidden_dim"], s["num_layers"], output_dim=s["output_dim"])
        adapter.load_state_dict(s["adapter_state"]); adapter.eval()
        norm = LabelNormalizer.from_dict(s["label_norm"])
        graphs, valid_idx = build_graphs(smiles)
        layer_embs = embed_per_layer(graphs, model, batch_size=kw.get("batch_size", 256))
        scalers = [(np.array(mu), np.array(sig)) for mu, sig in s["layer_scalers"]]
        tensors = [torch.as_tensor((emb - mu) / sig, dtype=torch.float32)
                   for emb, (mu, sig) in zip(layer_embs, scalers)]
        mode = kw.get("mode", "ensemble")
        if mode == "early_exit":
            tol = kw.get("exit_tolerance", 0.05) / float(np.mean(norm.sigma))
            pred_norm, _ = adapter.predict_early_exit(tensors, tolerance=tol)
        else:
            pred_norm = adapter.predict_ensemble(tensors)
        return norm.inverse(pred_norm), valid_idx
