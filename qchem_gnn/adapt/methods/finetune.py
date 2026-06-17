# qchem_gnn/adapt/methods/finetune.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...graph import GraphBatch
from ...model import MolecularQuantumGNN
from ..backbone import build_graphs, load_backbone
from ..data import LabelNormalizer
from ..metrics import classification_metrics, regression_metrics
from .base import LoadedAdapter, MLPHead, TrainResult, make_loss, postprocess


class FinetuneMethod:
    name = "finetune"

    def train(self, backbone, model_config, data, train_idx, val_idx, test_idx, cfg) -> TrainResult:
        seed = int(cfg.training.get("seed", 42))
        torch.manual_seed(seed)
        model = backbone
        h_dim = model.encoder.atom_encoder.embedding_dim
        task = data.task
        y = data.targets
        T = y.shape[1]

        if task == "classification":
            norm = None
        else:
            norm = LabelNormalizer.fit(y[train_idx])

        def prep_y(arr):
            return arr if task == "classification" else norm.transform(arr)

        def eval_pred(raw):
            return postprocess(task, raw, norm)

        head = MLPHead(h_dim, output_dim=T,
                       hidden_dims=tuple(cfg.adapter.get("hidden_dims", (128, 64))),
                       dropout=float(cfg.adapter.get("dropout", 0.1)))

        epochs = int(cfg.training.get("epochs", 200))
        head_lr = float(cfg.training.get("head_lr", 1e-3))
        bb_lr = float(cfg.training.get("backbone_lr", 5e-5))
        bs = int(cfg.training.get("batch_size", 64))
        patience = int(cfg.training.get("patience", 30))
        grad_clip = float(cfg.training.get("grad_clip", 1.0))

        opt = torch.optim.Adam(
            [{"params": model.parameters(), "lr": bb_lr},
             {"params": head.parameters(), "lr": head_lr}], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        crit = make_loss(task)

        graphs_tr = [data.graphs[i] for i in train_idx]
        ytr = torch.as_tensor(prep_y(y[train_idx]), dtype=torch.float32)
        val_batch = GraphBatch.from_graphs([data.graphs[i] for i in val_idx]) if val_idx else None
        te_batch = GraphBatch.from_graphs([data.graphs[i] for i in test_idx]) if test_idx else None

        best_val = float("inf")
        best_model = {k: v.clone() for k, v in model.state_dict().items()}
        best_head = {k: v.clone() for k, v in head.state_dict().items()}
        wait = 0
        n = len(graphs_tr)
        for _ in range(epochs):
            model.train(); head.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                bidx = perm[s : s + bs]
                batch = GraphBatch.from_graphs([graphs_tr[i] for i in bidx.tolist()])
                opt.zero_grad()
                loss = crit(head(model.encode_graph_embeddings(batch)), ytr[bidx])
                loss.backward()
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
                opt.step()
            sched.step()
            if val_batch is not None:
                model.eval(); head.eval()
                with torch.no_grad():
                    raw_val = head(model.encode_graph_embeddings(val_batch)).cpu().numpy()
                vp = eval_pred(raw_val)
                if task == "classification":
                    val_score = 1.0 - classification_metrics(y[val_idx], vp)["auc"]
                else:
                    val_score = float(np.mean(np.abs(vp - y[val_idx])))
                if val_score < best_val:
                    best_val, wait = val_score, 0
                    best_model = {k: v.clone() for k, v in model.state_dict().items()}
                    best_head = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

        model.load_state_dict(best_model); head.load_state_dict(best_head)
        model.eval(); head.eval()
        test_metrics = {}
        if te_batch is not None:
            with torch.no_grad():
                raw_te = head(model.encode_graph_embeddings(te_batch)).cpu().numpy()
            tp = eval_pred(raw_te)
            if task == "classification":
                test_metrics = classification_metrics(y[test_idx], tp)
            else:
                test_metrics = regression_metrics(y[test_idx], tp)

        payload = {
            "adapter_type": "finetune",
            "task": task,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "head_state": head.state_dict(),
            "head_config": head.config(),
            "label_norm": None if norm is None else norm.to_dict(),
        }
        return TrainResult(payload=payload, test_metrics=test_metrics,
                           log={"best_val_score": round(best_val, 4)})

    def save(self, path: Path, result: TrainResult, meta: dict) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**result.payload, **meta,
                    "training_info": {**meta.get("training_info", {}),
                                      "test_metrics": result.test_metrics, **result.log}}, path)

    @staticmethod
    def load(path: Path) -> LoadedAdapter:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return LoadedAdapter(adapter_type="finetune", payload=state)

    @staticmethod
    def predict(loaded: LoadedAdapter, smiles: list[str], **kw) -> tuple[np.ndarray, list[int]]:
        s = loaded.payload
        task = s.get("task", "regression")
        model = MolecularQuantumGNN(**s["model_config"])
        model.load_state_dict(s["model_state"]); model.eval()
        head = MLPHead(**s["head_config"]); head.load_state_dict(s["head_state"]); head.eval()
        norm = None if task == "classification" else LabelNormalizer.from_dict(s["label_norm"])
        graphs, valid_idx = build_graphs(smiles)
        bs = kw.get("batch_size", 256)
        chunks = []
        for st in range(0, len(graphs), bs):
            batch = GraphBatch.from_graphs(graphs[st : st + bs])
            with torch.no_grad():
                chunks.append(head(model.encode_graph_embeddings(batch)).cpu().numpy())
        if chunks:
            raw = np.concatenate(chunks)
            preds = postprocess(task, raw, norm)
        else:
            n_targets = s["head_config"]["output_dim"]
            preds = np.empty((0, n_targets))
        return preds, valid_idx
