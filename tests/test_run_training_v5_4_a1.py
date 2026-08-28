"""
HiMoFlow v5.4 — synthetic test for run_training_v5_4_a1.py (Batch 3).

Tests the training-loop machinery on a tiny synthetic dataset:
  - EMA decay updates correctly (state_dict roundtrip works)
  - EMA warmup tracks recent params at low step counts
  - cosine LR schedule produces sensible values at known checkpoints
  - LOSS_KEYS filter renames F→F_mat / L→L_mat and strips metadata
  - one full train_a1 call: writes checkpoints, history, config
  - resume from latest.pt picks up where it left off

Intentionally NOT testing convergence — that's a real-data concern.
Run with:
    python tests/test_run_training_v5_4_a1.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile

import numpy as np
import torch

# Make package importable
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from run_training_v5_4_a1 import (
    EMA, cosine_lr_with_warmup, _filter_batch_to_loss_inputs,
    LOSS_KEYS, train_a1,
)
from meanflow.ring_layout_diffusion import (
    build_ring_layout_diffusion, R_MAX, P_MAX,
)


def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")


def assert_close(a, b, msg, tol=1e-5):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: expected ~{b}, got {a}")


# ─── Test 1: EMA basic ─────────────────────────────────────────────────

def test_ema_basic():
    print("test_ema_basic ...")
    model = build_ring_layout_diffusion(capacity="600K")
    ema = EMA(model, decay=0.99)

    # Modify a param, then update EMA
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    assert_eq(ema.step, 1, "ema step incremented")

    backup = ema.apply_to(model)
    n_changed = sum(
        1 for n, p in model.named_parameters()
        if (p - backup[n]).abs().sum() > 1e-9
    )
    assert n_changed > 0, "applying EMA should change at least one param"

    ema.restore_to(model, backup)
    for n, p in model.named_parameters():
        if n in backup:
            assert (p - backup[n]).abs().sum() < 1e-9, \
                f"restore failed for {n}"
    print("  PASS")


# ─── Test 2: EMA warmup ────────────────────────────────────────────────

def test_ema_warmup():
    """The min(decay, (step+1)/(step+10)) trick must override decay
    for early steps. At step 1, eff = min(0.999, 2/11) = 0.18.
    Shadow = 0.18 * old + 0.82 * new ≈ closer to new."""
    print("test_ema_warmup ...")
    model = build_ring_layout_diffusion(capacity="600K")
    ema = EMA(model, decay=0.999)

    p_init = list(model.parameters())[0].clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ema.update(model)

    p_after = list(model.parameters())[0]
    shadow = list(ema.shadow.values())[0]
    d_to_after = (shadow - p_after).abs().mean()
    d_to_init = (shadow - p_init).abs().mean()
    assert d_to_after < d_to_init, \
        "warmup should make first-step EMA track current weights closely"
    print("  PASS")


# ─── Test 3: EMA state-dict roundtrip ──────────────────────────────────

def test_ema_state_dict_roundtrip():
    """Save EMA state, load into a fresh EMA, verify shadow is preserved."""
    print("test_ema_state_dict_roundtrip ...")
    model = build_ring_layout_diffusion(capacity="600K")
    ema = EMA(model, decay=0.99)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    ema.update(model)
    ema.update(model)
    sd = ema.state_dict()

    ema2 = EMA(model, decay=0.99)  # different shadow
    ema2.load_state_dict(sd, device=torch.device("cpu"))
    assert_eq(ema2.step, ema.step, "step preserved")
    for k in ema.shadow:
        assert (ema2.shadow[k] - ema.shadow[k]).abs().sum() < 1e-9, \
            f"shadow[{k}] not preserved across roundtrip"
    print("  PASS")


# ─── Test 4: cosine LR schedule sanity ─────────────────────────────────

def test_cosine_lr_schedule():
    print("test_cosine_lr_schedule ...")
    base = 3e-4
    warmup = 100
    total = 1000

    # Step 0: just starting, lr = base * 1/100
    lr0 = cosine_lr_with_warmup(0, warmup, total, base)
    assert_close(lr0, base / warmup, "lr at step 0", tol=1e-7)

    # End of warmup: lr ≈ base
    lr_warm = cosine_lr_with_warmup(warmup, warmup, total, base)
    assert_close(lr_warm, base, "lr at end of warmup", tol=1e-7)

    # End: cosine(pi) = -1, lr = base * 0.1
    lr_end = cosine_lr_with_warmup(total, warmup, total, base)
    assert_close(lr_end, base * 0.1, "lr at end (min_lr_frac=0.1)", tol=1e-7)

    # Beyond total: clamped at min_lr
    lr_beyond = cosine_lr_with_warmup(total + 100, warmup, total, base)
    assert_close(lr_beyond, base * 0.1, "lr clamped past total", tol=1e-7)
    print("  PASS")


# ─── Test 5: LOSS_KEYS filter renames F→F_mat, L→L_mat, strips meta ────

def test_loss_keys_filter_rename_and_strip():
    print("test_loss_keys_filter_rename_and_strip ...")
    fake_batch = {
        "R":         torch.zeros(4, R_MAX, dtype=torch.long),
        "F":         torch.zeros(4, R_MAX, R_MAX, dtype=torch.long),
        "L":         torch.zeros(4, R_MAX, R_MAX, dtype=torch.long),
        "P_len":     torch.zeros(4, R_MAX, P_MAX, dtype=torch.long),
        "P_pos":     torch.zeros(4, R_MAX, P_MAX, dtype=torch.long),
        "condition": torch.zeros(4, 2),
        # Metadata that should be stripped:
        "smi":       ["c1ccccc1"] * 4,
        "M_total":   torch.zeros(4, dtype=torch.long),
        "atom_ids":  torch.zeros(4, 6, dtype=torch.long),
    }
    filtered = _filter_batch_to_loss_inputs(fake_batch)
    assert_eq(set(filtered.keys()), set(LOSS_KEYS), "filtered keys")

    # Critical rename checks
    assert "F" not in filtered, "F should be renamed to F_mat"
    assert "L" not in filtered, "L should be renamed to L_mat"
    assert "F_mat" in filtered, "F_mat key missing"
    assert "L_mat" in filtered, "L_mat key missing"

    # Metadata stripped
    for k in ("smi", "M_total", "atom_ids"):
        assert k not in filtered, f"metadata key '{k}' should be stripped"
    print("  PASS")


def test_loss_keys_filter_missing_raises():
    print("test_loss_keys_filter_missing_raises ...")
    incomplete = {"R": torch.zeros(4, R_MAX, dtype=torch.long)}
    try:
        _filter_batch_to_loss_inputs(incomplete)
        raise AssertionError("should have raised KeyError")
    except KeyError as e:
        assert "missing" in str(e).lower(), f"wrong error message: {e}"
    print("  PASS")


# ─── Test 6: end-to-end train + resume ─────────────────────────────────

def _make_synthetic_label():
    """Build one synthetic A1 label resembling a 2-fused-ring layout
    with random condition."""
    R = np.zeros(R_MAX, dtype=np.int64)
    R[0] = 1; R[1] = 1
    F_ = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    F_[0, 1] = 1; F_[1, 0] = 1
    L_ = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    P_len = np.zeros((R_MAX, P_MAX), dtype=np.int64)
    P_pos = np.zeros((R_MAX, P_MAX), dtype=np.int64)
    cond = np.random.randn(2).astype(np.float32)
    return {
        "smi": "c1ccc2ccccc2c1",  # naphthalene
        "scaffold_smi": "c1ccc2ccccc2c1",
        "R": R, "F": F_, "L": L_, "P_len": P_len, "P_pos": P_pos,
        "atom_ids": np.ones(10, dtype=np.int64),
        "M_total": 10,
        "terminals": [],
        "condition": cond,
    }


def test_train_and_resume_smoke():
    """Build a 30-mol synthetic dataset, train 2 epochs, verify
    checkpoints/history/config are written, then resume for 1 more
    epoch and verify state continuity."""
    print("test_train_and_resume_smoke ...")
    np.random.seed(0)
    labels = [_make_synthetic_label() for _ in range(30)]

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f:
            pickle.dump(labels, f)

        # Train 2 epochs
        train_a1(
            labels_pkl_path=pkl_path,
            ckpt_dir=ckpt_dir,
            num_epochs=2,
            batch_size=4,
            capacity="600K",
            lr=1e-3,
            warmup_steps=2,
            val_fraction=0.2,
            seed=0,
            log_every_n_steps=0,
            eval_every_n_epochs=1,
            eval_n_samples=4,
            eval_n_steps=4,
        )
        for f in ("latest.pt", "best_model.pt", "ema.pt",
                  "history.json", "config.json"):
            p = os.path.join(ckpt_dir, f)
            assert os.path.exists(p), f"missing {f}"

        with open(os.path.join(ckpt_dir, "history.json"), "r") as f:
            history = json.load(f)
        assert_eq(len(history), 2, "history length after 2 epochs")
        # Each epoch record has the keys we expect
        for k in ("epoch", "train_loss", "val_loss", "lr",
                  "time_sec", "train_metrics", "val_metrics",
                  "decode_rate", "decode_rejection_reasons"):
            assert k in history[0], f"history record missing '{k}'"

        # Resume — train 1 more epoch
        train_a1(
            labels_pkl_path=pkl_path,
            ckpt_dir=ckpt_dir,
            num_epochs=3,
            batch_size=4,
            capacity="600K",
            lr=1e-3,
            warmup_steps=2,
            val_fraction=0.2,
            seed=0,
            log_every_n_steps=0,
            eval_every_n_epochs=1,
            eval_n_samples=4,
            eval_n_steps=4,
        )
        with open(os.path.join(ckpt_dir, "history.json"), "r") as f:
            history2 = json.load(f)
        assert_eq(len(history2), 3,
                  "history length after resume + 1 more epoch")
        # The first 2 entries should be identical (preserved from prior run)
        for i in (0, 1):
            assert_eq(history2[i]["train_loss"], history[i]["train_loss"],
                      f"epoch {i} train_loss preserved across resume")
            assert_eq(history2[i]["val_loss"], history[i]["val_loss"],
                      f"epoch {i} val_loss preserved across resume")

        # best_val_loss in latest.pt must match min val_loss across both
        # runs combined (resume should not reset or stale-overwrite it).
        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                        map_location="cpu", weights_only=False)
        all_val_losses = [h["val_loss"] for h in history2]
        assert_close(ck["best_val_loss"], min(all_val_losses),
                     "latest.pt best_val_loss matches min val across all epochs",
                     tol=1e-6)

    print("  PASS")


# ─── Test 7: condition_dim auto-detection (1-D regression) ─────────────

def test_condition_dim_autodetect_1d():
    """User's labels file had 1-element conditions (e.g., solubility
    only) where the model defaulted to condition_dim=2. train_a1 must
    auto-detect from the first label's condition shape and pass that
    to build_ring_layout_diffusion. Regression test for the
    "(B, 65) × (66, 192) shape mismatch" bug.
    """
    print("test_condition_dim_autodetect_1d ...")
    np.random.seed(0)
    labels = []
    for _ in range(20):
        lab = _make_synthetic_label()
        lab["condition"] = np.array([np.random.uniform(0.01, 0.87)],
                                     dtype=np.float32)  # 1-D!
        labels.append(lab)

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f:
            pickle.dump(labels, f)

        # Should not raise — model should auto-build with condition_dim=1.
        train_a1(
            labels_pkl_path=pkl_path,
            ckpt_dir=ckpt_dir,
            num_epochs=1,
            batch_size=4,
            capacity="600K",
            lr=1e-3,
            warmup_steps=2,
            val_fraction=0.2,
            seed=0,
            log_every_n_steps=0,
            eval_n_samples=2,
            eval_n_steps=2,
        )

        # The saved checkpoint's cond_in_proj.0.weight should have
        # input dim = 1 + 64 (time_embed_dim) = 65.
        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                        map_location="cpu", weights_only=False)
        cond_w = ck["model"]["cond_in_proj.0.weight"]
        # Linear weight shape is (out_features, in_features)
        assert_eq(cond_w.shape[1], 1 + 64,
                  "cond_in_proj input dim should be condition_dim(1) + time_embed_dim(64)")
    print("  PASS")


# ─── Driver ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 — synthetic test for run_training_v5_4_a1.py")
    print("=" * 70)
    test_ema_basic()
    test_ema_warmup()
    test_ema_state_dict_roundtrip()
    test_cosine_lr_schedule()
    test_loss_keys_filter_rename_and_strip()
    test_loss_keys_filter_missing_raises()
    test_train_and_resume_smoke()
    test_condition_dim_autodetect_1d()
    print()
    print("All 8 training-script tests PASSED")


if __name__ == "__main__":
    main()
