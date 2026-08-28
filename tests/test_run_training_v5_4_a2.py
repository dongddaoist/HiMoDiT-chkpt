"""
HiMoFlow v5.4 — synthetic test for run_training_v5_4_a2.py (Batch 4).

Tests training-loop machinery against a tiny synthetic dataset:
  - LOSS_KEYS filter strips metadata, raises on missing keys
  - EMA basics + state-dict roundtrip
  - cosine LR schedule sanity
  - A2ScaffoldDataset produces correctly-shaped tensors via the decoder
  - end-to-end train + resume smoke

Intentionally NOT testing convergence — that's a real-data concern.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path: sys.path.insert(0, PARENT)

from run_training_v5_4_a2 import (
    EMA, cosine_lr_with_warmup, _filter_batch_to_loss_inputs,
    LOSS_KEYS, train_a2, A2ScaffoldDataset, evaluate_sample_quality,
)
from meanflow.ring_atom_diffusion import (
    build_ring_atom_diffusion, M_MAX, N_ATOM_CLASSES,
)


def assert_eq(a, b, msg):
    if a != b: raise AssertionError(f"{msg}: expected {b}, got {a}")


def assert_close(a, b, msg, tol=1e-5):
    if abs(a - b) > tol: raise AssertionError(f"{msg}: expected ~{b}, got {a}")


# ─── Test 1: LOSS_KEYS filter ──────────────────────────────────────

def test_loss_keys_filter():
    print("test_loss_keys_filter ...")
    fake = {
        "atom_ids":     torch.zeros(4, M_MAX, dtype=torch.long),
        "bond_classes": torch.zeros(4, M_MAX, M_MAX, dtype=torch.long),
        "atom_mask":    torch.ones(4, M_MAX, dtype=torch.bool),
        "arom_mask":    torch.zeros(4, M_MAX, dtype=torch.bool),
        "condition":    torch.zeros(4, 2),
        # Metadata that should be stripped:
        "smi": ["c1ccccc1"] * 4,
        "M_total": torch.zeros(4, dtype=torch.long),
    }
    out = _filter_batch_to_loss_inputs(fake)
    assert_eq(set(out.keys()), set(LOSS_KEYS), "filtered keys")
    for k in ("smi", "M_total"):
        assert k not in out, f"metadata '{k}' not stripped"
    print("  PASS")


def test_loss_keys_filter_missing_raises():
    print("test_loss_keys_filter_missing_raises ...")
    incomplete = {"atom_ids": torch.zeros(4, M_MAX, dtype=torch.long)}
    try:
        _filter_batch_to_loss_inputs(incomplete)
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "missing" in str(e).lower(), f"wrong msg: {e}"
    print("  PASS")


# ─── Test 2: EMA ──────────────────────────────────────────────────

def test_ema_basic():
    print("test_ema_basic ...")
    model = build_ring_atom_diffusion(capacity="1M")
    ema = EMA(model, decay=0.99)
    with torch.no_grad():
        for p in model.parameters(): p.add_(torch.randn_like(p))
    ema.update(model)
    assert_eq(ema.step, 1, "ema step")
    backup = ema.apply_to(model)
    n_changed = sum(1 for n, p in model.named_parameters()
                     if (p - backup[n]).abs().sum() > 1e-9)
    assert n_changed > 0, "applying EMA should change params"
    ema.restore_to(model, backup)
    for n, p in model.named_parameters():
        if n in backup:
            assert (p - backup[n]).abs().sum() < 1e-9, f"restore failed for {n}"
    print("  PASS")


def test_ema_state_dict_roundtrip():
    print("test_ema_state_dict_roundtrip ...")
    model = build_ring_atom_diffusion(capacity="1M")
    ema = EMA(model, decay=0.99)
    with torch.no_grad():
        for p in model.parameters(): p.add_(torch.randn_like(p))
    ema.update(model); ema.update(model)
    sd = ema.state_dict()
    ema2 = EMA(model, decay=0.99)
    ema2.load_state_dict(sd, device=torch.device("cpu"))
    assert_eq(ema2.step, ema.step, "step preserved")
    for k in ema.shadow:
        assert (ema2.shadow[k] - ema.shadow[k]).abs().sum() < 1e-9, \
            f"shadow[{k}] not preserved"
    print("  PASS")


# ─── Test 3: cosine LR ─────────────────────────────────────────────

def test_cosine_lr_schedule():
    print("test_cosine_lr_schedule ...")
    base, warmup, total = 3e-4, 100, 1000
    assert_close(cosine_lr_with_warmup(0, warmup, total, base),
                  base/warmup, "step 0", tol=1e-7)
    assert_close(cosine_lr_with_warmup(warmup, warmup, total, base),
                  base, "end of warmup", tol=1e-7)
    assert_close(cosine_lr_with_warmup(total, warmup, total, base),
                  base*0.1, "end of training", tol=1e-7)
    print("  PASS")


# ─── Test 4: dataset produces correct shapes ──────────────────────

def _make_synthetic_label():
    """Naphthalene-shaped layout."""
    R = np.array([1, 1, 0, 0], dtype=np.int64)
    F_ = np.zeros((4, 4), dtype=np.int64); F_[0, 1] = 1; F_[1, 0] = 1
    L_ = np.zeros((4, 4), dtype=np.int64)
    P_len = np.zeros((4, 4), dtype=np.int64)
    P_pos = np.zeros((4, 4), dtype=np.int64)
    return {
        "smi": "c1ccc2ccccc2c1",
        "scaffold_smi": "c1ccc2ccccc2c1",
        "R": R, "F": F_, "L": L_, "P_len": P_len, "P_pos": P_pos,
        "atom_ids": np.ones(10, dtype=np.int64),  # 10 aromatic c
        "M_total": 10,
        "terminals": [],
        "condition": np.array([0.5, 0.6], dtype=np.float32),
    }


def test_dataset_shapes_via_decoder():
    print("test_dataset_shapes_via_decoder ...")
    labels = [_make_synthetic_label() for _ in range(3)]
    ds = A2ScaffoldDataset(labels)
    item = ds[0]
    assert_eq(item["atom_ids"].shape, (M_MAX,), "atom_ids shape")
    assert_eq(item["bond_classes"].shape, (M_MAX, M_MAX), "bond_classes shape")
    assert_eq(item["atom_mask"].shape, (M_MAX,), "atom_mask shape")
    assert_eq(item["arom_mask"].shape, (M_MAX,), "arom_mask shape")
    assert_eq(item["condition"].shape, (2,), "condition shape")
    # Naphthalene has 10 atoms; first 10 of atom_mask and arom_mask should be True
    assert item["atom_mask"][:10].all().item(), "first 10 atoms should be valid"
    assert (~item["atom_mask"][10:]).all().item(), "atoms 10+ should be padding"
    assert item["arom_mask"][:10].all().item(), "all naphthalene atoms aromatic"
    # Bonds: should have aromatic bonds (class=2) somewhere in the upper-left 10x10
    n_arom_bonds = (item["bond_classes"][:10, :10] == 2).sum().item() // 2
    assert_eq(n_arom_bonds, 11, "naphthalene has 11 aromatic bonds")
    print("  PASS")


# ─── Test 5: end-to-end train + resume ───────────────────────────

def test_train_and_resume_smoke():
    print("test_train_and_resume_smoke ...")
    np.random.seed(0)
    labels = [_make_synthetic_label() for _ in range(30)]
    # Vary conditions slightly
    for i, lab in enumerate(labels):
        lab["condition"] = np.array(
            [0.5 + 0.01 * i, 0.6 + 0.005 * i], dtype=np.float32,
        )

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f: pickle.dump(labels, f)

        train_a2(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=2, batch_size=4, capacity="1M", lr=1e-3,
            warmup_steps=2, val_fraction=0.2, seed=0,
            log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=2, eval_n_steps=4,
        )
        for f in ("latest.pt", "best_model.pt", "ema.pt",
                  "history.json", "config.json"):
            assert os.path.exists(os.path.join(ckpt_dir, f)), f"missing {f}"

        with open(os.path.join(ckpt_dir, "history.json")) as f:
            history = json.load(f)
        assert_eq(len(history), 2, "history len after 2 epochs")

        for k in ("epoch", "train_loss", "val_loss", "sample_atom_acc",
                  "sample_arom_compliance"):
            assert k in history[0], f"history missing {k}"

        # Aromatic compliance should be 100% (constraint enforced)
        assert_close(history[-1]["sample_arom_compliance"], 1.0,
                     "arom compliance must be 100%", tol=1e-6)

        # Resume + 1 more epoch
        train_a2(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=3, batch_size=4, capacity="1M", lr=1e-3,
            warmup_steps=2, val_fraction=0.2, seed=0,
            log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=2, eval_n_steps=4,
        )
        with open(os.path.join(ckpt_dir, "history.json")) as f:
            history2 = json.load(f)
        assert_eq(len(history2), 3, "history len after resume")
        for i in (0, 1):
            assert_eq(history2[i]["train_loss"], history[i]["train_loss"],
                      f"epoch {i} train_loss preserved")
            assert_eq(history2[i]["val_loss"], history[i]["val_loss"],
                      f"epoch {i} val_loss preserved")

        # best_val_loss in latest.pt matches min across all 3 epochs
        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                         map_location="cpu", weights_only=False)
        all_val = [h["val_loss"] for h in history2]
        assert_close(ck["best_val_loss"], min(all_val),
                     "best_val_loss matches min across run", tol=1e-6)
    print("  PASS")


def test_condition_dim_autodetect_2d():
    """A2 must auto-detect 2-D conditioning (sol, gap) and build the
    model with the right cond_in_proj shape."""
    print("test_condition_dim_autodetect_2d ...")
    np.random.seed(0)
    labels = [_make_synthetic_label() for _ in range(20)]
    # 2-D condition (sol, gap)
    for lab in labels:
        lab["condition"] = np.random.rand(2).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f: pickle.dump(labels, f)

        train_a2(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=1, batch_size=4, capacity="1M", lr=1e-3,
            warmup_steps=2, val_fraction=0.2, seed=0,
            log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=2, eval_n_steps=2,
        )
        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                         map_location="cpu", weights_only=False)
        cond_w = ck["model"]["cond_in_proj.0.weight"]
        # Linear weight shape (out, in); in = condition_dim + time_embed_dim(64)
        assert_eq(cond_w.shape[1], 2 + 64, "2-D cond + 64 time_embed")
    print("  PASS")


# ─── Driver ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 — synthetic test for run_training_v5_4_a2.py")
    print("=" * 70)
    test_loss_keys_filter()
    test_loss_keys_filter_missing_raises()
    test_ema_basic()
    test_ema_state_dict_roundtrip()
    test_cosine_lr_schedule()
    test_dataset_shapes_via_decoder()
    test_train_and_resume_smoke()
    test_condition_dim_autodetect_2d()
    print()
    print("All 8 training-script tests PASSED")


if __name__ == "__main__":
    main()
