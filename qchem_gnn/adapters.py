"""
Molecular property adaptation for pretrained GNN backbones.

Two adaptation strategies, both saving to a .pt file with a common
adapter_type key so predict_property.py and load_adapter() work
identically regardless of method:

  mlp_head  — backbone frozen; configurable MLP trained on fixed
               final embeddings; fast (embeddings computed once)

  finetune  — backbone + MLP head updated jointly with separate
               learning rates; slower but can recover task-specific
               features the pretraining representation missed

The ENGINE side-structure adapter lives in engine_adapter.py and
saves adapter_type="engine"; all three types are dispatchable through
the unified predict() function at the bottom of this module.

Public API
----------
MLPHead                    nn.Module used by both adapters
embed_smiles               final-layer GNN embeddings for a SMILES list
train_mlp_head             train frozen-backbone MLP, return state
train_finetune             train backbone+head jointly, return state
save_mlp_adapter           serialise MLP adapter to .pt
load_mlp_adapter           deserialise MLP adapter
save_finetune_adapter      serialise fine-tune checkpoint to .pt
load_finetune_adapter      deserialise fine-tune checkpoint
predict                    end-to-end SMILES → prediction (any adapter)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint
from .graph import GraphBatch, GraphData, build_graph_from_smiles
from .model import MolecularQuantumGNN


# ---------------------------------------------------------------------------
# Shared MLP head architecture
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Feedforward regression head: Linear → ReLU → Dropout, repeated for
    each hidden layer, then a final Linear to a scalar.

    Parameters
    ----------
    input_dim    dimension of the incoming feature vector
    hidden_dims  sequence of hidden-layer widths (default: (128, 64))
    dropout      dropout probability applied after each ReLU (default 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
        self.input_dim  = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout     = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {
            "input_dim":   self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout":     self.dropout,
        }


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def embed_smiles(
    smiles_list: list[str],
    model: MolecularQuantumGNN,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract the final frozen GNN embedding for each parseable SMILES.

    Returns
    -------
    embeddings  [N_valid, hidden_dim] float32 array
    valid_idx   list of indices into smiles_list that parsed successfully
    """
    valid_idx: list[int] = []
    graphs: list[GraphData] = []
    for i, smi in enumerate(smiles_list):
        try:
            graphs.append(build_graph_from_smiles(smi))
            valid_idx.append(i)
        except Exception:
            pass

    model.eval()
    chunks: list[np.ndarray] = []
    for s in range(0, len(graphs), batch_size):
        batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
        with torch.no_grad():
            chunks.append(model.encode_graph_embeddings(batch).cpu().numpy())
    emb = np.concatenate(chunks) if chunks else np.empty((0, model.encoder.atom_encoder.embedding_dim))
    return emb, valid_idx


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff   = pred - true
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "mae":  float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "r2":   (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def _print_header(method_name: str, epochs: int) -> None:
    print(f"\nTraining {method_name}  ({epochs} epochs) …")
    print(f"  {'Epoch':>6}  {'train_loss':>11}  {'val_MAE':>8}")
    print("  " + "-" * 32)


def _print_epoch(epoch: int, loss: float, val_mae: float) -> None:
    print(f"  {epoch:>6}  {loss:>11.4f}  {val_mae:>8.4f}")


def _print_test(name: str, m: dict[str, float], W: int = 32) -> None:
    print(f"  {name:<{W}}  {m['mae']:6.3f}  {m['rmse']:6.3f}  {m['r2']:6.3f}")


# ---------------------------------------------------------------------------
# MLP head adapter  (backbone frozen)
# ---------------------------------------------------------------------------

def train_mlp_head(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.1,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    patience: int = 40,
    seed: int = 42,
    print_every: int = 50,
) -> tuple[MLPHead, np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    """
    Train an MLPHead on pre-computed frozen embeddings.

    Feature and label normalisation is fit on the training split only.

    Returns
    -------
    head          trained MLPHead (best-val checkpoint)
    feat_mu       [D] feature mean  (store for inference)
    feat_sig      [D] feature std   (store for inference)
    label_mu      label mean
    label_sig     label std
    log           dict with loss/val_mae histories and test metrics
    """
    torch.manual_seed(seed)

    # Normalisation statistics fit on training data only
    feat_mu  = X_tr.mean(0)
    feat_sig = X_tr.std(0).clip(1e-8)
    label_mu  = float(y_tr.mean())
    label_sig = float(y_tr.std()) or 1.0

    def _nf(X): return (X - feat_mu) / feat_sig
    def _ny(y): return (y - label_mu) / label_sig
    def _dy(y): return y * label_sig + label_mu

    X_tr_t  = torch.as_tensor(_nf(X_tr),  dtype=torch.float32)
    X_val_t = torch.as_tensor(_nf(X_val), dtype=torch.float32)
    X_te_t  = torch.as_tensor(_nf(X_te),  dtype=torch.float32)
    y_tr_t  = torch.as_tensor(_ny(y_tr),  dtype=torch.float32)

    head  = MLPHead(X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    opt   = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    crit  = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    wait       = 0
    loss_hist: list[float] = []
    val_hist:  list[float] = []
    n = len(X_tr_t)

    if print_every > 0:
        _print_header(f"MLP head  [hidden={list(hidden_dims)}, dropout={dropout}]", epochs)

    for epoch in range(1, epochs + 1):
        head.train()
        perm  = torch.randperm(n)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            loss = crit(head(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        loss_hist.append(total / n)

        head.eval()
        with torch.no_grad():
            val_pred = _dy(head(X_val_t).numpy())
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        val_hist.append(val_mae)

        if val_mae < best_val:
            best_val   = val_mae
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if print_every > 0:
                    print(f"  Early stop at epoch {epoch}  (best val MAE {best_val:.4f})")
                break

        if print_every > 0 and (epoch % print_every == 0 or epoch == epochs):
            _print_epoch(epoch, loss_hist[-1], val_mae)

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_pred = _dy(head(X_te_t).numpy())

    log = {
        "best_val_mae":   round(best_val, 4),
        "loss_history":   loss_hist,
        "val_mae_history": val_hist,
        **{f"test_{k}": round(v, 4) for k, v in _metrics(test_pred, y_te).items()},
    }
    return head, feat_mu, feat_sig, label_mu, label_sig, log


def save_mlp_adapter(
    path: Path | str,
    head: MLPHead,
    feat_mu: np.ndarray,
    feat_sig: np.ndarray,
    label_mu: float,
    label_sig: float,
    backbone_ckpt: Path | str,
    training_info: dict[str, Any] | None = None,
) -> None:
    """Serialise MLP head adapter and normalisation stats to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_type":  "mlp_head",
            "head_state":    head.state_dict(),
            "head_config":   head.config(),
            "feat_mu":       feat_mu.tolist(),
            "feat_sig":      feat_sig.tolist(),
            "label_mu":      float(label_mu),
            "label_sig":     float(label_sig),
            "backbone_ckpt": str(backbone_ckpt),
            "training_info": training_info or {},
        },
        path,
    )


def load_mlp_adapter(
    path: Path | str,
) -> tuple[MLPHead, dict[str, Any]]:
    """
    Load a saved MLP head adapter.

    Returns (head, meta) where meta contains feat_mu, feat_sig,
    label_mu, label_sig, backbone_ckpt, and training_info.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg   = state["head_config"]
    head  = MLPHead(cfg["input_dim"], hidden_dims=cfg["hidden_dims"], dropout=cfg.get("dropout", 0.1))
    head.load_state_dict(state["head_state"])
    head.eval()
    meta: dict[str, Any] = {
        "feat_mu":       np.array(state["feat_mu"]),
        "feat_sig":      np.array(state["feat_sig"]),
        "label_mu":      state["label_mu"],
        "label_sig":     state["label_sig"],
        "backbone_ckpt": state["backbone_ckpt"],
        "training_info": state.get("training_info", {}),
    }
    return head, meta


# ---------------------------------------------------------------------------
# Fine-tune adapter  (backbone + head updated jointly)
# ---------------------------------------------------------------------------

def train_finetune(
    model: MolecularQuantumGNN,
    graphs_tr: list[GraphData],
    y_tr: np.ndarray,
    graphs_val: list[GraphData],
    y_val: np.ndarray,
    graphs_te: list[GraphData],
    y_te: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.1,
    epochs: int = 200,
    head_lr: float = 1e-3,
    backbone_lr: float = 5e-5,
    batch_size: int = 64,
    patience: int = 30,
    grad_clip: float = 1.0,
    seed: int = 42,
    print_every: int = 50,
) -> tuple[MolecularQuantumGNN, MLPHead, float, float, dict[str, Any]]:
    """
    Fine-tune the GNN backbone and a fresh MLP head jointly.

    The backbone uses a much lower learning rate than the head to
    preserve the pretraining representation while adapting it to
    the downstream task.  Gradient clipping prevents exploding
    gradients through the deeper backbone layers.

    Returns
    -------
    model      fine-tuned backbone (best-val checkpoint)
    head       trained MLP head (best-val checkpoint)
    label_mu   label mean
    label_sig  label std
    log        dict with loss/val histories and test metrics
    """
    torch.manual_seed(seed)

    h_dim = model.encoder.atom_encoder.embedding_dim
    head  = MLPHead(h_dim, hidden_dims=hidden_dims, dropout=dropout)

    label_mu  = float(y_tr.mean())
    label_sig = float(y_tr.std()) or 1.0

    def _ny(y): return (y - label_mu) / label_sig
    def _dy(y): return y * label_sig + label_mu

    y_tr_t = torch.as_tensor(_ny(y_tr), dtype=torch.float32)

    # Two parameter groups: backbone at lower LR, head at higher LR
    opt = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": backbone_lr},
            {"params": head.parameters(),  "lr": head_lr},
        ],
        weight_decay=1e-5,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )
    crit = nn.MSELoss()

    # Pre-build static validation and test batches
    val_batch = GraphBatch.from_graphs(graphs_val)
    te_batch  = GraphBatch.from_graphs(graphs_te)

    best_val    = float("inf")
    best_model  = None
    best_head   = None
    wait        = 0
    loss_hist:  list[float] = []
    val_hist:   list[float] = []
    n = len(graphs_tr)

    if print_every > 0:
        _print_header(
            f"fine-tune  [backbone_lr={backbone_lr}, head_lr={head_lr}, "
            f"hidden={list(hidden_dims)}, dropout={dropout}]",
            epochs,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        perm  = torch.randperm(n)
        total = 0.0

        for s in range(0, n, batch_size):
            batch_idx = perm[s : s + batch_size]
            batch = GraphBatch.from_graphs([graphs_tr[i] for i in batch_idx.tolist()])
            opt.zero_grad()
            emb  = model.encode_graph_embeddings(batch)
            loss = crit(head(emb), y_tr_t[batch_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), grad_clip
            )
            opt.step()
            total += loss.item() * len(batch_idx)

        sched.step()
        loss_hist.append(total / n)

        model.eval()
        head.eval()
        with torch.no_grad():
            val_pred = _dy(head(model.encode_graph_embeddings(val_batch)).cpu().numpy())
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        val_hist.append(val_mae)

        if val_mae < best_val:
            best_val   = val_mae
            best_model = {k: v.clone() for k, v in model.state_dict().items()}
            best_head  = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if print_every > 0:
                    print(f"  Early stop at epoch {epoch}  (best val MAE {best_val:.4f})")
                break

        if print_every > 0 and (epoch % print_every == 0 or epoch == epochs):
            _print_epoch(epoch, loss_hist[-1], val_mae)

    model.load_state_dict(best_model)
    head.load_state_dict(best_head)
    model.eval()
    head.eval()

    with torch.no_grad():
        test_pred = _dy(head(model.encode_graph_embeddings(te_batch)).cpu().numpy())

    log = {
        "best_val_mae":    round(best_val, 4),
        "loss_history":    loss_hist,
        "val_mae_history": val_hist,
        **{f"test_{k}": round(v, 4) for k, v in _metrics(test_pred, y_te).items()},
    }
    return model, head, label_mu, label_sig, log


def save_finetune_adapter(
    path: Path | str,
    model: MolecularQuantumGNN,
    head: MLPHead,
    label_mu: float,
    label_sig: float,
    model_config: dict[str, Any],
    backbone_ckpt: Path | str,
    training_info: dict[str, Any] | None = None,
) -> None:
    """Serialise fine-tuned backbone + head to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_type":  "finetune",
            "model_state":   model.state_dict(),
            "model_config":  model_config,
            "head_state":    head.state_dict(),
            "head_config":   head.config(),
            "label_mu":      float(label_mu),
            "label_sig":     float(label_sig),
            "backbone_ckpt": str(backbone_ckpt),
            "training_info": training_info or {},
        },
        path,
    )


def load_finetune_adapter(
    path: Path | str,
) -> tuple[MolecularQuantumGNN, MLPHead, dict[str, Any]]:
    """
    Load a saved fine-tune adapter.

    Returns (model, head, meta) where meta contains label_mu, label_sig,
    backbone_ckpt, training_info, and model_config.
    """
    state  = torch.load(Path(path), map_location="cpu", weights_only=False)
    model  = MolecularQuantumGNN(**state["model_config"])
    model.load_state_dict(state["model_state"])
    cfg    = state["head_config"]
    head   = MLPHead(cfg["input_dim"], hidden_dims=cfg["hidden_dims"], dropout=cfg.get("dropout", 0.1))
    head.load_state_dict(state["head_state"])
    model.eval()
    head.eval()
    meta: dict[str, Any] = {
        "label_mu":      state["label_mu"],
        "label_sig":     state["label_sig"],
        "backbone_ckpt": state["backbone_ckpt"],
        "training_info": state.get("training_info", {}),
        "model_config":  state["model_config"],
    }
    return model, head, meta


# ---------------------------------------------------------------------------
# Unified end-to-end prediction  (all adapter types)
# ---------------------------------------------------------------------------

def predict(
    smiles_list: list[str],
    adapter_path: Path | str,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    """
    Predict a property for a list of SMILES using any saved adapter.

    Dispatches on the adapter_type key:  mlp_head | finetune | engine

    Returns
    -------
    predictions  [N_valid] float32 array in original label units
    valid_idx    indices into smiles_list that were successfully parsed
    """
    adapter_path = Path(adapter_path)
    header = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter_type = header.get("adapter_type", "engine")

    if adapter_type == "mlp_head":
        head, meta = load_mlp_adapter(adapter_path)
        ckpt  = load_checkpoint(Path(meta["backbone_ckpt"]))
        model = MolecularQuantumGNN(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        emb, valid_idx = embed_smiles(smiles_list, model, batch_size)
        X = torch.as_tensor(
            (emb - np.array(meta["feat_mu"])) / np.array(meta["feat_sig"]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred = head(X).numpy() * meta["label_sig"] + meta["label_mu"]
        return pred, valid_idx

    if adapter_type == "finetune":
        model, head, meta = load_finetune_adapter(adapter_path)
        valid_idx: list[int] = []
        graphs: list[GraphData] = []
        for i, smi in enumerate(smiles_list):
            try:
                graphs.append(build_graph_from_smiles(smi))
                valid_idx.append(i)
            except Exception:
                pass
        preds: list[np.ndarray] = []
        for s in range(0, len(graphs), batch_size):
            batch = GraphBatch.from_graphs(graphs[s : s + batch_size])
            with torch.no_grad():
                preds.append(head(model.encode_graph_embeddings(batch)).cpu().numpy())
        pred_arr = (
            np.concatenate(preds) * meta["label_sig"] + meta["label_mu"]
            if preds else np.empty(0)
        )
        return pred_arr, valid_idx

    if adapter_type == "engine":
        from .engine_adapter import predict as engine_predict
        backbone = header.get("backbone_checkpoint") or header.get("backbone_ckpt")
        return engine_predict(smiles_list, backbone, adapter_path, batch_size=batch_size)

    raise ValueError(f"Unknown adapter_type: {adapter_type!r}")
