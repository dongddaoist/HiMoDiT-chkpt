"""
HiMoFlow v5.4 — synthetic test for run_training_v5_4_terminal.py (Batch 5).

Tests:
  - LOSS_KEYS filter strips metadata, raises on missing keys
  - EMA basics + state-dict roundtrip
  - cosine LR schedule sanity
  - TerminalDataset produces correct tensor shapes via the decoder
  - missing host_canonical_idx raises a clear error
  - Class-weights computation from labels
  - end-to-end train + resume smoke with synthetic naphthalene-with-OH set
  - condition_dim auto-detect (2-D)
  - v5.3 warm-start through training script

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

from run_training_v5_4_terminal import (
    EMA, cosine_lr_with_warmup, _filter_batch_to_loss_inputs,
    LOSS_KEYS, train_terminal, TerminalDataset,
    compute_class_weights_from_labels,
    evaluate_sample_quality,
)
from meanflow.terminal_fragment_diffusion import (
    build_fragment_stage2, V5_3_NUM_FRAGMENTS, DEFAULT_NUM_FRAGMENTS,
)
from meanflow.ring_layout_decoder import M_MAX


def assert_eq(a, b, msg):
    if a != b: raise AssertionError(f"{msg}: expected {b}, got {a}")
def assert_close(a, b, msg, tol=1e-5):
    if abs(a - b) > tol: raise AssertionError(f"{msg}: expected ~{b}, got {a}")


# ─── Test 1: LOSS_KEYS filter ──────────────────────────────────────

def test_loss_keys_filter():
    print("test_loss_keys_filter ...")
    fake = {
        "scaffold_atom_ids":     torch.zeros(4, M_MAX, dtype=torch.long),
        "scaffold_bond_classes": torch.zeros(4, M_MAX, M_MAX, dtype=torch.long),
        "scaffold_atom_mask":    torch.ones(4, M_MAX, dtype=torch.bool),
        "site_fragment_ids":     torch.zeros(4, M_MAX, dtype=torch.long),
        "condition":             torch.zeros(4, 2),
        "smi":                   ["c1ccccc1"] * 4,   # metadata
        "M_total":               torch.zeros(4, dtype=torch.long),
    }
    out = _filter_batch_to_loss_inputs(fake)
    assert_eq(set(out.keys()), set(LOSS_KEYS), "filtered keys")
    for k in ("smi", "M_total"):
        assert k not in out, f"metadata '{k}' not stripped"
    print("  PASS")


def test_loss_keys_missing_raises():
    print("test_loss_keys_missing_raises ...")
    incomplete = {"scaffold_atom_ids": torch.zeros(4, M_MAX, dtype=torch.long)}
    try:
        _filter_batch_to_loss_inputs(incomplete)
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "missing" in str(e).lower(), f"wrong msg: {e}"
    print("  PASS")


# ─── Test 2: EMA ──────────────────────────────────────────────────

def test_ema_basic():
    print("test_ema_basic ...")
    model = build_fragment_stage2(capacity="1M")
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
            assert (p - backup[n]).abs().sum() < 1e-9, f"restore failed {n}"
    print("  PASS")


def test_ema_state_dict_roundtrip():
    print("test_ema_state_dict_roundtrip ...")
    model = build_fragment_stage2(capacity="1M")
    ema = EMA(model, decay=0.99)
    with torch.no_grad():
        for p in model.parameters(): p.add_(torch.randn_like(p))
    ema.update(model); ema.update(model)
    sd = ema.state_dict()
    ema2 = EMA(model, decay=0.99)
    ema2.load_state_dict(sd, device=torch.device("cpu"))
    assert_eq(ema2.step, ema.step, "step preserved")
    for k in ema.shadow:
        assert (ema2.shadow[k] - ema.shadow[k]).abs().sum() < 1e-9
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


# ─── Test 4: dataset ──────────────────────────────────────────────

def _make_synthetic_label(with_terminal=True):
    """Naphthalene layout, optionally with one OH terminal at canonical
    slot 3."""
    R = np.array([1, 1, 0, 0], dtype=np.int64)
    F_ = np.zeros((4, 4), dtype=np.int64); F_[0, 1] = 1; F_[1, 0] = 1
    L_ = np.zeros((4, 4), dtype=np.int64)
    P_len = np.zeros((4, 4), dtype=np.int64)
    P_pos = np.zeros((4, 4), dtype=np.int64)
    label = {
        "smi": "c1ccc2ccccc2c1",
        "scaffold_smi": "c1ccc2ccccc2c1",
        "R": R, "F": F_, "L": L_, "P_len": P_len, "P_pos": P_pos,
        "atom_ids": np.ones(10, dtype=np.int64),
        "M_total": 10,
        "terminals": ([
            {
                "name": "OH",
                "atom_indices": [99],
                "host_atom_idx": 7,
                "host_canonical_idx": 3,
                "anchor_atom_idx": 99,
                "attach_bond_class": 1,
                "host_is_aromatic": True,
            }
        ] if with_terminal else []),
        "condition": np.array([0.5, 0.6], dtype=np.float32),
    }
    return label


def test_dataset_shapes_and_site_marking():
    print("test_dataset_shapes_and_site_marking ...")
    labels = [_make_synthetic_label(with_terminal=True) for _ in range(3)]
    ds = TerminalDataset(labels, num_fragments=DEFAULT_NUM_FRAGMENTS)
    item = ds[0]
    assert_eq(item["scaffold_atom_ids"].shape, (M_MAX,), "atom_ids shape")
    assert_eq(item["scaffold_bond_classes"].shape, (M_MAX, M_MAX), "bonds")
    assert_eq(item["scaffold_atom_mask"].shape, (M_MAX,), "atom_mask")
    assert_eq(item["site_fragment_ids"].shape, (M_MAX,), "site_ids")
    assert_eq(item["condition"].shape, (2,), "condition shape")
    # Slot 3 should be marked OH = SMARTS-id 0 → model class 1
    assert_eq(item["site_fragment_ids"][3].item(), 1,
              "OH at slot 3 should map to model class 1")
    # Other valid slots should be 0 (no decoration)
    for s in range(10):
        if s == 3: continue
        assert_eq(item["site_fragment_ids"][s].item(), 0,
                  f"slot {s} should be 0")
    # Padding should be 0
    assert (item["site_fragment_ids"][10:] == 0).all().item(), \
            "padding should be 0"
    print("  PASS")


def test_dataset_missing_canonical_idx_raises():
    print("test_dataset_missing_canonical_idx_raises ...")
    bad = _make_synthetic_label(with_terminal=True)
    del bad["terminals"][0]["host_canonical_idx"]
    try:
        TerminalDataset([bad])
        raise AssertionError("should have raised")
    except KeyError as e:
        assert "host_canonical_idx" in str(e), f"wrong msg: {e}"
    print("  PASS")


def test_dataset_skips_out_of_vocab_terminal():
    """If num_fragments=6 (v5.3 vocab) but a label has =O (model class 7),
    that terminal should be silently skipped, leaving site=0."""
    print("test_dataset_skips_out_of_vocab_terminal ...")
    lab = _make_synthetic_label(with_terminal=True)
    lab["terminals"][0] = {
        "name": "=O", "atom_indices": [99], "host_atom_idx": 7,
        "host_canonical_idx": 4, "anchor_atom_idx": 99,
        "attach_bond_class": 3, "host_is_aromatic": False,
    }
    ds = TerminalDataset([lab], num_fragments=V5_3_NUM_FRAGMENTS)
    item = ds[0]
    # =O has SMARTS-id 6 → model class 7 → out of K=6 model's range → skip
    assert_eq(item["site_fragment_ids"][4].item(), 0,
              "=O should be skipped when num_fragments=6")
    print("  PASS")


# ─── Test 5: class weights from labels ─────────────────────────────

def test_class_weights_from_labels():
    print("test_class_weights_from_labels ...")
    # 3 labels, each with one OH; total 30 atoms, 3 are decorated (OH).
    labels = [_make_synthetic_label(with_terminal=True) for _ in range(3)]
    w, counts = compute_class_weights_from_labels(
        labels, num_fragments=DEFAULT_NUM_FRAGMENTS,
    )
    assert_eq(w.shape, (10,), "weights shape")
    assert_eq(counts[0], 27, "counts[no_decoration]: 30-3=27")
    assert_eq(counts[1], 3, "counts[OH]: 3")
    # Class 0 (most common) has lower weight than class 1
    assert w[0] < w[1], "class 0 should weight less than class 1"
    # Mean weight ~= 1
    assert abs(float(w.mean().item()) - 1.0) < 1e-5, \
            "weights should normalize to mean=1"
    print("  PASS")


# ─── Test 6: end-to-end train + resume ───────────────────────────

def test_train_and_resume_smoke():
    print("test_train_and_resume_smoke ...")
    np.random.seed(0)
    labels = [_make_synthetic_label(with_terminal=(i % 2 == 0))
              for i in range(40)]
    for i, lab in enumerate(labels):
        lab["condition"] = np.array(
            [0.5 + 0.01*i, 0.6 + 0.005*i], dtype=np.float32,
        )

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f: pickle.dump(labels, f)

        train_terminal(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=2, batch_size=4, capacity="1M",
            num_fragments=DEFAULT_NUM_FRAGMENTS,
            lr=1e-3, warmup_steps=2, val_fraction=0.2,
            seed=0, log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=4, eval_n_steps=4,
        )
        for f in ("latest.pt", "best_model.pt", "ema.pt",
                  "history.json", "config.json"):
            assert os.path.exists(os.path.join(ckpt_dir, f)), f"missing {f}"

        with open(os.path.join(ckpt_dir, "history.json")) as f:
            hist = json.load(f)
        assert_eq(len(hist), 2, "history len after 2 epochs")

        for k in ("epoch", "train_loss", "val_loss",
                   "sample_atom_acc", "sample_baseline_acc"):
            assert k in hist[0], f"history missing {k}"

        # Resume + 1 more epoch
        train_terminal(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=3, batch_size=4, capacity="1M",
            num_fragments=DEFAULT_NUM_FRAGMENTS,
            lr=1e-3, warmup_steps=2, val_fraction=0.2,
            seed=0, log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=4, eval_n_steps=4,
        )
        with open(os.path.join(ckpt_dir, "history.json")) as f:
            hist2 = json.load(f)
        assert_eq(len(hist2), 3, "history len after resume")
        for i in (0, 1):
            assert_eq(hist2[i]["train_loss"], hist[i]["train_loss"],
                      f"epoch {i} train_loss preserved")

        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                         map_location="cpu", weights_only=False)
        all_val = [h["val_loss"] for h in hist2]
        assert_close(ck["best_val_loss"], min(all_val),
                     "best_val_loss matches min", tol=1e-6)
    print("  PASS")


def test_condition_dim_autodetect_2d():
    print("test_condition_dim_autodetect_2d ...")
    np.random.seed(0)
    labels = [_make_synthetic_label() for _ in range(20)]
    for lab in labels:
        lab["condition"] = np.random.rand(2).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f: pickle.dump(labels, f)

        train_terminal(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=1, batch_size=4, capacity="1M",
            num_fragments=DEFAULT_NUM_FRAGMENTS,
            lr=1e-3, warmup_steps=2, val_fraction=0.2,
            seed=0, log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=2, eval_n_steps=2,
        )
        ck = torch.load(os.path.join(ckpt_dir, "latest.pt"),
                         map_location="cpu", weights_only=False)
        cond_w = ck["model"]["cond_in_proj.weight"]
        # Linear weight shape (out, in); in = condition_dim + time_embed_dim(64)
        assert_eq(cond_w.shape[1], 2 + 64, "2-D cond + 64 time_embed")
    print("  PASS")


# ─── Test 7: warm-start through training script ────────────────────

def test_train_with_v5_3_warmstart():
    """Pre-train a fake K=6 'v5.3' model, save it, then run train_terminal
    with --init-from pointing at it. Verify rows 0..6 of frag_head are
    initialized from the v5.3 checkpoint."""
    print("test_train_with_v5_3_warmstart ...")
    np.random.seed(0)
    # Synthetic labels with extended-vocab terminal (so the new
    # heads see signal)
    labels = []
    for i in range(20):
        lab = _make_synthetic_label(with_terminal=True)
        if i % 4 == 0:
            # Replace OH with =O occasionally (model class 7)
            lab["terminals"][0]["name"] = "=O"
            lab["terminals"][0]["attach_bond_class"] = 3
        labels.append(lab)

    with tempfile.TemporaryDirectory() as td:
        v53_path = os.path.join(td, "v53_best.pt")
        # Build a K=6 model and save its state_dict
        m_v53 = build_fragment_stage2(capacity="1M",
                                       num_fragments=V5_3_NUM_FRAGMENTS)
        with torch.no_grad():
            for p in m_v53.parameters():
                p.fill_(0.5)
        torch.save(m_v53.state_dict(), v53_path)

        pkl_path = os.path.join(td, "labels.pkl")
        ckpt_dir = os.path.join(td, "ckpt")
        with open(pkl_path, "wb") as f: pickle.dump(labels, f)

        result = train_terminal(
            labels_pkl_path=pkl_path, ckpt_dir=ckpt_dir,
            num_epochs=1, batch_size=4, capacity="1M",
            num_fragments=DEFAULT_NUM_FRAGMENTS,
            lr=0.0,   # no learning; verify warm-start preserves weights
            warmup_steps=2, val_fraction=0.2,
            seed=0, log_every_n_steps=0,
            eval_every_n_epochs=1, eval_n_samples=2, eval_n_steps=2,
            init_from=v53_path,
        )
        status = result["warmstart_status"]
        assert status is not None, "warm-start status not returned"
        assert status["frag_head.weight"] == "expanded"
        assert status["frag_head.bias"] == "expanded"
        assert status["frag_input_embed.weight"] == "expanded"
        assert status["atom_embed.weight"] == "transferred"

        # With lr=0, the saved best_model should still have the v5.3
        # weights for IDs 0..6 (modulo nothing, since lr=0)
        best = torch.load(os.path.join(ckpt_dir, "best_model.pt"),
                          map_location="cpu", weights_only=False)
        fh_w = best["frag_head.weight"]   # (10, d_model)
        # Rows 0..6 should equal v5.3's filled value 0.5
        assert torch.allclose(fh_w[:7], torch.full_like(fh_w[:7], 0.5),
                               atol=1e-5), \
                "warm-started v5.3 weights got perturbed"
        # Rows 7..9 should be zero
        assert torch.allclose(fh_w[7:], torch.zeros_like(fh_w[7:]),
                               atol=1e-5), \
                "new-class rows should remain zero with lr=0"
    print("  PASS")


# ─── Driver ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 — synthetic test for run_training_v5_4_terminal.py")
    print("=" * 70)
    test_loss_keys_filter()
    test_loss_keys_missing_raises()
    test_ema_basic()
    test_ema_state_dict_roundtrip()
    test_cosine_lr_schedule()
    test_dataset_shapes_and_site_marking()
    test_dataset_missing_canonical_idx_raises()
    test_dataset_skips_out_of_vocab_terminal()
    test_class_weights_from_labels()
    test_train_and_resume_smoke()
    test_condition_dim_autodetect_2d()
    test_train_with_v5_3_warmstart()
    print()
    print("All 12 training-script tests PASSED")


if __name__ == "__main__":
    main()
