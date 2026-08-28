"""
HiMoFlow v5.5 — A2 (atom-assignment diffusion) training script.
==================================================================

Trains the RingAtomDiffusion model on v5.5 labels produced by
extract_layout_v5_5(). A2 is conditioned on the clean bond skeleton
(produced by decode_v5_5_to_scaffold) and the property condition; it
predicts atom IDs in the K=16 v5.5 vocab.

Differences from v5.4 training script:
  - Imports decode_v5_5_to_scaffold + aromatic_constraint_mask_v5_5
    instead of v5.4's build_bond_classes/aromatic_constraint_mask
  - Dataset uses v5.5 fields (B_size/B_pos/B_parent/B_bond + spiro_atom_positions)
    instead of v5.4's P_len/P_pos
  - A2 model file (meanflow/ring_atom_diffusion.py) is already v5.5-shaped
    (N_ATOM_CLASSES=16); no model edits required

Mirrors A1/A3/Terminal conventions:
  - EMA, cosine LR with warmup, auto-resume from latest.pt
  - Saves: latest.pt, best_model.pt, ema.pt, history.json, config.json
  - Per-epoch one-line summary (mid-step logging off by default)
  - Per-epoch sample-quality eval on EMA weights

Use as a script:
    python run_training_v5_5_a2.py \\
        --labels-pkl ./dataset_labels_v5_5.pkl \\
        --ckpt-dir   ./checkpoints_a2_v5_5 \\
        --epochs 40 --batch-size 256 --capacity 10M
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meanflow.ring_atom_diffusion import (
    build_ring_atom_diffusion, count_parameters,
    M_MAX, N_ATOM_CLASSES, AROMATIC_ATOM_IDS, ATOM_PAD,
)
from meanflow.ring_layout_decoder import (
    decode_v5_5_to_scaffold, aromatic_constraint_mask_v5_5,
)


# v5.5 bond vocab: K=5 (was K=3 in v5.3/v5.4 RedDB; encoder now produces
# BOND_DOUBLE=3 and BOND_TRIPLE=4 in addition to none/single/aromatic).
# The A2 model's N_BOND_CLASSES defaults to 3; must override to 5 or
# F.one_hot(bond_classes, num_classes=3) hits indices >= 3 and triggers
# a CUDA assert in _build_edge_probs.
V5_5_NUM_BOND_CLASSES = 5


# ─── Loss-input filter ──────────────────────────────────────────────

LOSS_KEYS = [
    "atom_ids", "bond_classes", "atom_mask", "arom_mask", "condition",
]


def _filter_batch_to_loss_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Strip dataset metadata; pass through loss-relevant tensors."""
    out = {k: batch[k] for k in LOSS_KEYS if k in batch}
    missing = set(LOSS_KEYS) - set(out.keys())
    if missing:
        raise KeyError(f"batch missing required keys for compute_loss: {missing}")
    return out


# ─── EMA ────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average with warmup-aware effective decay.
    Mirrors run_training_v5_4_a1.EMA exactly."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.step = 0
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.step += 1
        eff = min(self.decay, (self.step + 1) / (self.step + 10))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                self.shadow[n].mul_(eff).add_(p.detach(), alpha=1.0 - eff)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        return backup

    def restore_to(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup: p.copy_(backup[n])

    def state_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "decay": self.decay,
                "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd: Dict[str, Any], device: torch.device) -> None:
        self.step = sd["step"]
        self.decay = sd["decay"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}


# ─── LR schedule ────────────────────────────────────────────────────

def cosine_lr_with_warmup(step, warmup_steps, total_steps, base_lr, min_lr_frac=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


# ─── Dataset ───────────────────────────────────────────────────────

class A2ScaffoldDataset(Dataset):
    """Wraps v5.5 layout labels. Each __getitem__ runs the v5.5 decoder
    to produce (atom_ids_padded, bond_classes, atom_mask, arom_mask) on
    the fly. This avoids storing a precomputed redundant representation;
    the labels file stays compact.

    The decoder pass for a single 24-atom scaffold is fast (~0.1 ms),
    well under the GPU forward time per batch element.
    """
    def __init__(self, labels: List[Dict[str, Any]]):
        # Sanity-check v5.5 schema on first label
        if labels:
            required_v5_5 = ("B_size", "B_pos", "B_parent", "B_bond",
                              "spiro_atom_positions")
            lab0 = labels[0]
            missing = [k for k in required_v5_5 if k not in lab0]
            if missing:
                raise KeyError(
                    f"Labels missing v5.5 fields {missing}. "
                    f"Re-run preprocessing with extract_layout_v5_5()."
                )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lab = self.labels[idx]
        R  = np.asarray(lab["R"], dtype=np.int64)
        F_ = np.asarray(lab["F"], dtype=np.int64)
        L_ = np.asarray(lab["L"], dtype=np.int64)
        B_size   = np.asarray(lab["B_size"],   dtype=np.int64)
        B_pos    = np.asarray(lab["B_pos"],    dtype=np.int64)
        B_parent = np.asarray(lab["B_parent"], dtype=np.int64)
        B_bond   = np.asarray(lab["B_bond"],   dtype=np.int64)
        spiro_atom_positions = np.asarray(
            lab["spiro_atom_positions"], dtype=np.int64,
        )
        atom_ids_compact = np.asarray(lab["atom_ids"], dtype=np.int64)
        M_total = int(lab["M_total"])

        # Pad atom_ids to M_MAX
        atom_ids_padded = np.zeros(M_MAX, dtype=np.int64)
        atom_ids_padded[:M_total] = atom_ids_compact

        # v5.5 decoder produces bond_classes + atom_mask consistent with
        # the v5.5 encoder's canonical traversal (branches included).
        bond_classes, atom_mask = decode_v5_5_to_scaffold(
            R, F_, L_, B_size, B_pos, B_parent, B_bond,
            spiro_atom_positions, atom_ids_padded, M_MAX_out=M_MAX,
        )
        arom_mask = aromatic_constraint_mask_v5_5(bond_classes, atom_mask)

        cond = torch.as_tensor(lab["condition"], dtype=torch.float32).reshape(-1)
        return {
            "atom_ids":     torch.from_numpy(atom_ids_padded).long(),
            "bond_classes": torch.from_numpy(bond_classes).long(),
            "atom_mask":    torch.from_numpy(atom_mask).bool(),
            "arom_mask":    torch.from_numpy(arom_mask).bool(),
            "condition":    cond,
        }


def _collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([it[k] for it in items], dim=0) for k in items[0]}


# ─── Sample-quality eval ────────────────────────────────────────────

@torch.no_grad()
def evaluate_sample_quality(
    model: nn.Module,
    eval_dataset: Dataset,
    n_samples: int = 64,
    n_steps: int = 20,
    cfg_scale: float = 1.0,
    seed: int = 12345,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Sample atom IDs given REAL bond skeletons drawn from the eval set.

    For each sample:
      1. Pull a (bond_classes, atom_mask, arom_mask, condition) from eval set.
      2. Run A2.sample to get atom_ids.
      3. Try decode_layout_to_scaffold round-trip — should always pass
         because the aromatic constraint is enforced in sampling.
      4. Compute argmax-vs-truth atom-ID accuracy as a bonus signal.
    """
    model.eval()
    n_eval = min(n_samples, len(eval_dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eval_dataset), size=n_eval, replace=False)
    items = [eval_dataset[int(i)] for i in indices]
    batch = {k: torch.stack([it[k] for it in items], dim=0).to(device) for k in items[0]}

    samples = model.sample(
        bond_classes=batch["bond_classes"],
        atom_mask=batch["atom_mask"],
        arom_mask=batch["arom_mask"],
        condition=batch["condition"],
        n_steps=n_steps, cfg_scale=cfg_scale, seed=seed,
    )

    # Atom-ID accuracy on valid positions
    correct = (samples == batch["atom_ids"]) & batch["atom_mask"]
    denom = batch["atom_mask"].sum().clamp(min=1)
    atom_acc = (correct.sum().float() / denom).item()

    # Aromatic-constraint adherence (should be 100% by construction)
    arom_set = torch.tensor(AROMATIC_ATOM_IDS, device=device)
    pred_in_arom = (samples.unsqueeze(-1) == arom_set).any(dim=-1)
    valid_arom = batch["atom_mask"] & batch["arom_mask"]
    arom_compliance = (pred_in_arom[valid_arom].float().mean().item()
                        if valid_arom.any() else 1.0)

    # Aliphatic positions: fraction predicted in the aliphatic set (not strictly
    # required, but a useful health signal — if the model only ever picks
    # aromatic IDs at aliphatic positions, that's a problem).
    aliph_set = torch.tensor([2, 3, 4, 6, 7], device=device)
    pred_in_aliph = (samples.unsqueeze(-1) == aliph_set).any(dim=-1)
    valid_aliph = batch["atom_mask"] & (~batch["arom_mask"])
    aliph_pick_rate = (pred_in_aliph[valid_aliph].float().mean().item()
                        if valid_aliph.any() else float("nan"))

    return {
        "atom_acc": atom_acc,
        "arom_compliance": arom_compliance,
        "aliph_pick_rate": aliph_pick_rate,
        "n_evaluated": n_eval,
    }


# ─── Main training loop ─────────────────────────────────────────────

def train_a2(
    labels_pkl_path: str,
    ckpt_dir: str,
    num_epochs: int,
    batch_size: int = 128,
    capacity: str = "10M",
    lr: float = 3e-4,
    warmup_steps: int = 500,
    val_fraction: float = 0.05,
    seed: int = 0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    cfg_drop_prob: float = 0.1,
    edge_attn_enabled: bool = True,
    ema_decay: float = 0.999,
    eval_every_n_epochs: int = 1,
    eval_n_samples: int = 64,
    eval_n_steps: int = 20,
    num_workers: int = 0,
    device: Optional[str] = None,
    log_every_n_steps: int = 0,
) -> Dict[str, Any]:
    """Train A2 from labels_pkl_path. See module docstring for details."""
    os.makedirs(ckpt_dir, exist_ok=True)
    rng = random.Random(seed); np.random.seed(seed); torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[train_a2] device={device}, capacity={capacity}, "
          f"edge_attn_enabled={edge_attn_enabled}")

    # Load labels
    with open(labels_pkl_path, "rb") as f:
        labels = pickle.load(f)
    print(f"[train_a2] loaded {len(labels):,} labels from {labels_pkl_path}")

    rng.shuffle(labels)
    n_val = max(1, int(round(val_fraction * len(labels))))
    train_labels = labels[n_val:]
    val_labels = labels[:n_val]
    print(f"[train_a2] split train={len(train_labels):,}, val={len(val_labels):,}")

    # Auto-detect condition_dim
    sample_cond = np.asarray(labels[0]["condition"]).reshape(-1)
    detected_cond_dim = int(sample_cond.shape[0])
    print(f"[train_a2] detected condition_dim={detected_cond_dim} "
          f"(from labels[0]['condition']={sample_cond})")

    train_ds = A2ScaffoldDataset(train_labels)
    val_ds = A2ScaffoldDataset(val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
    )

    # Model
    model = build_ring_atom_diffusion(
        capacity=capacity, condition_dim=detected_cond_dim,
        cfg_drop_prob=cfg_drop_prob, edge_attn_enabled=edge_attn_enabled,
        n_bond_classes=V5_5_NUM_BOND_CLASSES,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[train_a2] model {capacity}: {n_params:,} params "
          f"(n_bond_classes={V5_5_NUM_BOND_CLASSES})")

    # Optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    print(f"[train_a2] steps/epoch={steps_per_epoch}, total_steps={total_steps}")

    ema = EMA(model, decay=ema_decay)

    # Auto-resume
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    history_path = os.path.join(ckpt_dir, "history.json")
    config_path = os.path.join(ckpt_dir, "config.json")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    ema_path = os.path.join(ckpt_dir, "ema.pt")

    start_epoch = 0; global_step = 0; best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    if os.path.exists(latest_path):
        print(f"[train_a2] resuming from {latest_path}")
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ema.load_state_dict(ck["ema"], device=device)
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_val_loss = ck["best_val_loss"]
        if os.path.exists(history_path):
            with open(history_path, "r") as f: history = json.load(f)
        print(f"[train_a2]  → resuming epoch {start_epoch} step {global_step} "
              f"best_val={best_val_loss:.4f}")

    if not os.path.exists(config_path):
        config = dict(
            labels_pkl_path=labels_pkl_path, num_epochs=num_epochs,
            batch_size=batch_size, capacity=capacity, lr=lr,
            warmup_steps=warmup_steps, val_fraction=val_fraction, seed=seed,
            weight_decay=weight_decay, grad_clip=grad_clip,
            cfg_drop_prob=cfg_drop_prob, edge_attn_enabled=edge_attn_enabled,
            n_bond_classes=V5_5_NUM_BOND_CLASSES,
            ema_decay=ema_decay, n_train=len(train_labels), n_val=len(val_labels),
            n_params=n_params, condition_dim=detected_cond_dim,
        )
        with open(config_path, "w") as f: json.dump(config, f, indent=2)

    # Training
    for epoch in range(start_epoch, num_epochs):
        model.train()
        t_start = time.time()
        epoch_losses: List[float] = []
        epoch_metrics: Dict[str, List[float]] = {}

        for step_in_epoch, batch in enumerate(train_loader):
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            inputs = _filter_batch_to_loss_inputs(batch_dev)

            cur_lr = cosine_lr_with_warmup(global_step, warmup_steps, total_steps, lr)
            for pg in opt.param_groups: pg["lr"] = cur_lr

            out = model.compute_loss(**inputs)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); ema.update(model)

            epoch_losses.append(loss.item())
            for k, v in out.items():
                if k == "loss": continue
                epoch_metrics.setdefault(k, []).append(float(v.item()))

            global_step += 1
            if log_every_n_steps > 0 and (step_in_epoch + 1) % log_every_n_steps == 0:
                print(f"  epoch {epoch:3d} step {step_in_epoch:5d}/{steps_per_epoch} "
                      f"| loss={loss.item():.4f} acc={out['acc'].item():.3f} "
                      f"lr={cur_lr:.2e}")

        train_loss = float(np.mean(epoch_losses))
        epoch_means = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}

        # Val eval
        model.eval()
        val_losses = []; val_metrics: Dict[str, List[float]] = {}
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                inputs = _filter_batch_to_loss_inputs(batch_dev)
                out = model.compute_loss(**inputs)
                val_losses.append(out["loss"].item())
                for k, v in out.items():
                    if k == "loss": continue
                    val_metrics.setdefault(k, []).append(float(v.item()))
        val_loss = float(np.mean(val_losses))
        val_means = {k: float(np.mean(v)) for k, v in val_metrics.items()}

        # Sample-quality eval (on EMA weights)
        eval_metrics = None
        if (epoch + 1) % eval_every_n_epochs == 0:
            backup = ema.apply_to(model)
            eval_metrics = evaluate_sample_quality(
                model, val_ds,
                n_samples=eval_n_samples, n_steps=eval_n_steps,
                seed=seed + epoch, device=device,
            )
            ema.restore_to(model, backup)

        record = dict(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss, lr=cur_lr,
            time_sec=time.time() - t_start,
            train_metrics=epoch_means, val_metrics=val_means,
        )
        if eval_metrics is not None:
            record["sample_atom_acc"] = eval_metrics["atom_acc"]
            record["sample_arom_compliance"] = eval_metrics["arom_compliance"]
            record["sample_aliph_pick_rate"] = eval_metrics["aliph_pick_rate"]
        history.append(record)

        # Save checkpoints (best updates BEFORE latest, mirroring B3 fix)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        torch.save(
            dict(model=model.state_dict(), optimizer=opt.state_dict(),
                 ema=ema.state_dict(), epoch=epoch, global_step=global_step,
                 best_val_loss=best_val_loss),
            latest_path,
        )
        torch.save(ema.state_dict(), ema_path)

        with open(history_path, "w") as f: json.dump(history, f, indent=2)

        # Per-epoch summary (one line, A1/A3/Terminal convention)
        def _g(d, k):
            return d.get(k, float("nan"))
        eval_str = ""
        if eval_metrics is not None:
            eval_str = (f" sample_acc={eval_metrics['atom_acc']*100:.1f}%"
                        f" arom={eval_metrics['arom_compliance']*100:.1f}%"
                        f" aliph={eval_metrics['aliph_pick_rate']*100:.1f}%")
        print(
            f"[A2][ep {epoch:3d}] "
            f"train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} "
            f"acc={_g(epoch_means, 'acc'):.3f}"
            f"{eval_str} "
            f"({record['time_sec']:.1f}s)",
            flush=True,
        )

    return dict(best_val_loss=best_val_loss, n_epochs=num_epochs, history=history)


# ─── CLI ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train HiMoFlow v5.5 A2.")
    p.add_argument("--labels-pkl", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--capacity", type=str, default="10M",
                    choices=["1M", "3M", "10M", "30M"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)
    p.add_argument("--edge-attn-enabled", action="store_true", default=True)
    p.add_argument("--no-edge-attn", dest="edge_attn_enabled", action="store_false",
                   help="Ablation: disable edge-biased attention")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--eval-n-samples", type=int, default=64)
    p.add_argument("--eval-n-steps", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every-n-steps", type=int, default=0,
                   help="If >0, also print a log line every N optimizer steps. "
                        "Default 0 = one summary line per epoch only.")
    return p.parse_args()


def _main():
    args = _parse_args()
    train_a2(
        labels_pkl_path=args.labels_pkl, ckpt_dir=args.ckpt_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        capacity=args.capacity, lr=args.lr,
        warmup_steps=args.warmup_steps, val_fraction=args.val_fraction,
        seed=args.seed, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, cfg_drop_prob=args.cfg_drop_prob,
        edge_attn_enabled=args.edge_attn_enabled,
        ema_decay=args.ema_decay,
        eval_every_n_epochs=args.eval_every_n_epochs,
        eval_n_samples=args.eval_n_samples, eval_n_steps=args.eval_n_steps,
        num_workers=args.num_workers, device=args.device,
        log_every_n_steps=args.log_every_n_steps,
    )


if __name__ == "__main__":
    _main()
