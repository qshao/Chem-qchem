# qchem_gnn/adapt/methods/mlp_head.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..backbone import build_graphs, embed_final, load_backbone
from ..data import LabelNormalizer
from ..metrics import classification_metrics, regression_metrics
from .base import LoadedAdapter, MLPHead, TrainResult, make_loss, postprocess


class MlpHeadMethod:
    name = "mlp_head"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)

        task = data.task
        emb = embed_final(data.graphs, backbone, batch_size=int(cfg.training.get("batch_size", 128)) * 2)
        y = data.targets
        T = y.shape[1]

        feat_mu = emb[train_idx].mean(0)
        feat_sig = emb[train_idx].std(0).clip(1e-8)

        if task == "classification":
            norm = None
        else:
            norm = LabelNormalizer.fit(y[train_idx])

        def nf(X): return (X - feat_mu) / feat_sig

        def prep_y(arr):
            return arr if task == "classification" else norm.transform(arr)

        def eval_pred(raw):
            return postprocess(task, raw, norm)

        Xtr = torch.as_tensor(nf(emb[train_idx]), dtype=torch.float32)
        Xva = torch.as_tensor(nf(emb[val_idx]), dtype=torch.float32) if val_idx else None
        Xte = torch.as_tensor(nf(emb[test_idx]), dtype=torch.float32) if test_idx else None
        ytr = torch.as_tensor(prep_y(y[train_idx]), dtype=torch.float32)

        head = MLPHead(emb.shape[1], output_dim=T,
                       hidden_dims=tuple(cfg.adapter.get("hidden_dims", (128, 64))),
                       dropout=float(cfg.adapter.get("dropout", 0.1)))
        epochs = int(cfg.training.get("epochs", 300))
        lr = float(cfg.training.get("head_lr", cfg.training.get("lr", 1e-3)))
        bs = int(cfg.training.get("batch_size", 128))
        patience = int(cfg.training.get("patience", 40))
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        crit = make_loss(task)

        best_val, best_state, wait = float("inf"), None, 0
        n = len(Xtr)
        for _ in range(epochs):
            head.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                opt.zero_grad()
                loss = crit(head(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            sched.step()
            if Xva is not None:
                head.eval()
                with torch.no_grad():
                    raw_val = head(Xva).numpy()
                vp = eval_pred(raw_val)
                if task == "classification":
                    val_score = 1.0 - classification_metrics(y[val_idx], vp)["auc"]
                else:
                    val_score = float(np.mean(np.abs(vp - y[val_idx])))
                if val_score < best_val:
                    best_val, wait = val_score, 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        if best_state is not None:
            head.load_state_dict(best_state)
        head.eval()
        test_metrics = {}
        if Xte is not None:
            with torch.no_grad():
                raw_te = head(Xte).numpy()
            tp = eval_pred(raw_te)
            if task == "classification":
                test_metrics = classification_metrics(y[test_idx], tp)
            else:
                test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "mlp_head",
            "task": task,
            "head_state": head.state_dict(),
            "head_config": head.config(),
            "feat_mu": feat_mu.tolist(),
            "feat_sig": feat_sig.tolist(),
            "label_norm": None if norm is None else norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_score": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta, "training_info": {**meta.get("training_info", {}),
                                                                "test_metrics": result.test_metrics,
                                                                **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="mlp_head", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        task = s.get("task", "regression")
        model, _ = load_backbone(s["backbone_ckpt"])
        graphs, valid_idx = build_graphs(smiles)
        emb = embed_final(graphs, model, batch_size=kw.get("batch_size", 256))
        cfg = s["head_config"]
        head = MLPHead(**cfg)
        head.load_state_dict(s["head_state"])
        head.eval()
        feat_mu = np.array(s["feat_mu"]); feat_sig = np.array(s["feat_sig"])
        norm = None if task == "classification" else LabelNormalizer.from_dict(s["label_norm"])
        with torch.no_grad():
            X = torch.as_tensor((emb - feat_mu) / feat_sig, dtype=torch.float32)
            raw = head(X).numpy()
        preds = postprocess(task, raw, norm)
        return preds, valid_idx
