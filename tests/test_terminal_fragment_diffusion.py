"""
HiMoFlow v5.4 — Unit tests for terminal_fragment_diffusion.py (Batch 5).

Tests cover:
  - Vocab/constants consistency with terminal_smarts_v3_extended
  - Corruption mechanics + alpha endpoints
  - Class weights computation
  - Model construction at all capacity presets
  - Forward shape contract
  - bias_enabled=False ablation path
  - Gradient flow through every param (2-step rule, same as B4)
  - Loss decrease on synthetic data
  - Initial loss within ±1.0 of log(K+1)
  - Padding contributes zero loss
  - Sample shape, vocab range, padding correctness
  - v5.3 warm-start: tensor names match v5.3, warm-start preserves
    weights for IDs 0-6 and zero-pads IDs 7-9
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from meanflow.terminal_fragment_diffusion import (
    NUM_BOND_CLASSES, DEFAULT_NUM_ATOM_TYPES, DEFAULT_NUM_FRAGMENTS,
    V5_3_NUM_FRAGMENTS, _mask_class_idx,
    corrupt_fragment_ids, compute_class_weights,
    FragmentStage2, build_fragment_stage2, count_parameters,
    load_v5_3_warmstart, CAPACITY_PRESETS,
)
from preprocessing.terminal_smarts_v3_extended import CURATED_TERMINALS


def _print_pass(name): print(f"  PASS  {name}")
def _assert(cond, msg):
    if not cond: raise AssertionError(msg)


# ─── Tests: vocab consistency ───────────────────────────────────────

def test_vocab_consistency_with_smarts_file():
    """The SMARTS file's v5_3_id should correspond to our model class
    via model_class = v5_3_id + 1, and total fragments K should match."""
    print("test_vocab_consistency_with_smarts_file")
    smarts_ids = sorted(t["v5_3_id"] for t in CURATED_TERMINALS)
    _assert(smarts_ids == list(range(len(CURATED_TERMINALS))),
            f"SMARTS ids not contiguous 0..{len(CURATED_TERMINALS)-1}: {smarts_ids}")
    _assert(len(CURATED_TERMINALS) == DEFAULT_NUM_FRAGMENTS,
            f"DEFAULT_NUM_FRAGMENTS={DEFAULT_NUM_FRAGMENTS} but SMARTS file has "
            f"{len(CURATED_TERMINALS)} entries")
    _assert(V5_3_NUM_FRAGMENTS == 6,
            f"V5_3_NUM_FRAGMENTS should be 6, got {V5_3_NUM_FRAGMENTS}")
    _print_pass("vocab_consistency_with_smarts_file")


def test_mask_class_idx():
    print("test_mask_class_idx")
    _assert(_mask_class_idx(6) == 7, "v5.3: K=6 → MASK=7")
    _assert(_mask_class_idx(9) == 10, "v5.4: K=9 → MASK=10")
    _print_pass("mask_class_idx")


# ─── Tests: corruption ───────────────────────────────────────────────

def test_corruption_padding_unchanged():
    print("test_corruption_padding_unchanged")
    torch.manual_seed(0)
    B, N = 4, 24
    fid = torch.randint(0, 7, (B, N))
    am = torch.zeros(B, N, dtype=torch.bool)
    am[:, :10] = True  # only first 10 valid
    alpha = torch.full((B,), 0.9)
    fid_t, is_masked = corrupt_fragment_ids(fid, alpha, am, num_fragments=9)
    # Padding never gets MASK
    mask_idx = _mask_class_idx(9)
    _assert(not (fid_t[:, 10:] == mask_idx).any(), "padding got MASK")
    _assert(not is_masked[:, 10:].any(), "is_masked True at padding")
    # ~90% of valid should be MASK
    rate = (fid_t[:, :10] == mask_idx).float().mean()
    _assert(0.7 < rate < 1.0, f"unexpected mask rate {rate:.3f}")
    _print_pass("corruption_padding_unchanged")


def test_corruption_alpha_zero():
    print("test_corruption_alpha_zero")
    fid = torch.tensor([[1, 3, 5, 0, 0]])
    am = torch.tensor([[True, True, True, False, False]])
    fid_t, is_masked = corrupt_fragment_ids(
        fid, torch.zeros(1), am, num_fragments=9,
    )
    _assert(torch.equal(fid_t, fid), "α=0 should leave all unchanged")
    _assert(not is_masked.any(), "α=0 should mask nothing")
    _print_pass("corruption_alpha_zero")


# ─── Tests: class weights ────────────────────────────────────────────

def test_class_weights_inverse_freq():
    print("test_class_weights_inverse_freq")
    counts = {0: 10000, 1: 100, 2: 200, 3: 50, 4: 0, 5: 75,
              6: 1, 7: 5, 8: 8, 9: 3}
    w = compute_class_weights(counts, num_fragments=9, smoothing=1.0)
    _assert(w.shape == (10,), f"weights shape {w.shape}")
    _assert(abs(float(w.mean().item()) - 1.0) < 1e-5,
            "weights should be normalized to mean=1")
    # Class 0 (most common) should have lowest weight; class 6 (rarest)
    # highest.
    _assert(w[0] < w[1], "common class weight should be lower")
    _assert(w[6] > w[0], "rare class weight should be higher")
    _print_pass("class_weights_inverse_freq")


# ─── Tests: model construction ───────────────────────────────────────

def test_param_counts_per_capacity():
    print("test_param_counts_per_capacity")
    expected_ranges = {
        "1M":  (700_000, 1_500_000),
        "3M":  (1_700_000, 3_500_000),
        "9M":  (7_000_000, 11_000_000),
        "30M": (28_000_000, 36_000_000),
    }
    for cap, (lo, hi) in expected_ranges.items():
        m = build_fragment_stage2(capacity=cap)
        n = count_parameters(m)
        _assert(lo <= n <= hi,
                f"{cap}: got {n:,} params, expected [{lo:,}, {hi:,}]")
        print(f"    {cap:>4}: {n:>12,} params")
    _print_pass("param_counts_per_capacity")


def test_unknown_capacity_raises():
    print("test_unknown_capacity_raises")
    try:
        build_fragment_stage2(capacity="99T")
        _assert(False, "should have raised")
    except ValueError as e:
        _assert("Unknown capacity" in str(e), f"wrong msg: {e}")
    _print_pass("unknown_capacity_raises")


# ─── Helpers ────────────────────────────────────────────────────────

def _make_dummy_batch(B=4, N=24, M=12, num_fragments=9, seed=0):
    """B molecules with M=12 valid scaffold atoms each."""
    torch.manual_seed(seed)
    sa = torch.zeros(B, N, dtype=torch.long)
    sa[:, :M] = torch.randint(1, DEFAULT_NUM_ATOM_TYPES, (B, M))
    am = torch.zeros(B, N, dtype=torch.bool); am[:, :M] = True
    bc = torch.zeros(B, N, N, dtype=torch.long)
    # Add a sparse aromatic bond pattern to exercise bond_bias
    for b in range(B):
        for i in range(M - 1):
            bc[b, i, i+1] = 2
            bc[b, i+1, i] = 2
    target = torch.zeros(B, N, dtype=torch.long)
    # ~30% of valid sites get a real fragment (1..K)
    rand = torch.rand(B, M)
    nonzero = rand < 0.3
    target[:, :M] = torch.where(
        nonzero,
        torch.randint(1, num_fragments + 1, (B, M)),
        torch.zeros(B, M, dtype=torch.long),
    )
    cond = torch.randn(B, 2)
    return dict(
        scaffold_atom_ids=sa, scaffold_bond_classes=bc,
        scaffold_atom_mask=am, site_fragment_ids=target,
        condition=cond,
    )


# ─── Tests: forward / loss / sample ──────────────────────────────────

def test_forward_shape():
    print("test_forward_shape")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.eval()
    batch = _make_dummy_batch(B=3, num_fragments=9)
    fid_t = torch.full_like(batch["site_fragment_ids"],
                             _mask_class_idx(9))
    alpha = torch.rand(3)
    logits = m(
        scaffold_atom_ids=batch["scaffold_atom_ids"],
        scaffold_bond_classes=batch["scaffold_bond_classes"],
        scaffold_atom_mask=batch["scaffold_atom_mask"],
        fragment_ids_t=fid_t, alpha=alpha,
        condition=batch["condition"],
    )
    _assert(logits.shape == (3, 24, 10),
            f"logits shape {logits.shape} != (3, 24, 10)")
    _print_pass("forward_shape")


def test_bias_disabled_path():
    print("test_bias_disabled_path")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M", bias_enabled=False)
    m.eval()
    batch = _make_dummy_batch(B=2, num_fragments=9)
    out = m.compute_loss(**batch)
    _assert(torch.isfinite(out["loss"]), "loss not finite for ablation")
    _assert(m.blocks[0].attn.bond_bias is None,
            "bond_bias should be None when bias_enabled=False")
    _print_pass("bias_disabled_path")


def test_gradient_flow_full():
    """All params accumulate gradient after 2 steps (AdaLN zero-init
    stalls cond_in_proj on step 0; resolved after one update)."""
    print("test_gradient_flow_full")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    init = {n: p.detach().clone() for n, p in m.named_parameters()}
    batch = _make_dummy_batch(B=4, num_fragments=9)
    for _ in range(2):
        out = m.compute_loss(**batch)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    n_changed = sum(
        1 for n, p in m.named_parameters()
        if (p.detach() - init[n]).abs().sum() > 1e-9
    )
    n_total = sum(1 for _ in m.parameters())
    _assert(n_changed == n_total,
            f"only {n_changed}/{n_total} params updated after 2 steps")
    _print_pass("gradient_flow_full")


def test_loss_decreases():
    print("test_loss_decreases")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    batch = _make_dummy_batch(B=8, num_fragments=9)
    losses = []
    for step in range(40):
        out = m.compute_loss(**batch)
        _assert(torch.isfinite(out["loss"]), f"NaN at step {step}")
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())
    _assert(losses[-1] < losses[0] * 0.5,
            f"loss didn't decrease: {losses[0]:.3f}→{losses[-1]:.3f}")
    print(f"    loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    _print_pass("loss_decreases")


def test_loss_initial_near_log_K():
    """Initial CE loss ~ log(K+1) = log(10) ≈ 2.30."""
    print("test_loss_initial_near_log_K")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.train()
    batch = _make_dummy_batch(B=32, num_fragments=9)
    out = m.compute_loss(**batch)
    expected = math.log(10)
    actual = out["loss"].item()
    _assert(abs(actual - expected) < 1.0,
            f"initial loss {actual:.3f} not near log(10)={expected:.3f}")
    _print_pass("loss_initial_near_log_K")


def test_padding_zero_loss_contribution():
    """Modifying padding-region fragment ids must not change loss."""
    print("test_padding_zero_loss_contribution")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.eval()
    batch = _make_dummy_batch(B=4, num_fragments=9)
    batch_b = {k: v.clone() if torch.is_tensor(v) else v
               for k, v in batch.items()}
    batch_b["site_fragment_ids"][:, 12:] = 5

    torch.manual_seed(42)
    out_a = m.compute_loss(**batch)
    torch.manual_seed(42)
    out_b = m.compute_loss(**batch_b)
    _assert(abs(out_a["loss"].item() - out_b["loss"].item()) < 1e-5,
            f"padding contributed to loss: {out_a['loss'].item()} vs "
            f"{out_b['loss'].item()}")
    _print_pass("padding_zero_loss_contribution")


def test_sample_shape_vocab_padding():
    print("test_sample_shape_vocab_padding")
    torch.manual_seed(0)
    m = build_fragment_stage2(capacity="1M"); m.eval()
    batch = _make_dummy_batch(B=4, num_fragments=9)
    samples = m.sample(
        scaffold_atom_ids=batch["scaffold_atom_ids"],
        scaffold_bond_classes=batch["scaffold_bond_classes"],
        scaffold_atom_mask=batch["scaffold_atom_mask"],
        condition=batch["condition"], n_steps=4, seed=42,
    )
    _assert(samples.shape == (4, 24), f"shape {samples.shape}")
    _assert((samples >= 0).all() and (samples <= 9).all(),
            "vocab range out of [0, 9]")
    _assert(samples.max().item() < _mask_class_idx(9),
            "MASK_IDX appeared in output")
    _assert((samples[:, 12:] == 0).all(),
            "padding produced non-zero predictions")
    _print_pass("sample_shape_vocab_padding")


# ─── Tests: v5.3 warm-start ─────────────────────────────────────────

def test_warmstart_tensor_names_match_v5_3():
    """The B5 model has identical tensor names to v5.3 (verify by
    constructing K=6 model and confirming key naming).

    v5.3 reference key set is the K=6 model's state_dict. The B5 model
    at K=9 must contain all the same keys (just with wider shapes for
    fragment-aware tensors)."""
    print("test_warmstart_tensor_names_match_v5_3")
    m_v53 = build_fragment_stage2(capacity="1M",
                                   num_fragments=V5_3_NUM_FRAGMENTS)
    m_b5  = build_fragment_stage2(capacity="1M",
                                   num_fragments=DEFAULT_NUM_FRAGMENTS)
    keys_v53 = set(m_v53.state_dict().keys())
    keys_b5  = set(m_b5.state_dict().keys())
    _assert(keys_v53 == keys_b5,
            f"v5.3 keys {sorted(keys_v53 - keys_b5)} missing from B5; "
            f"B5 extra keys: {sorted(keys_b5 - keys_v53)}")
    _print_pass("warmstart_tensor_names_match_v5_3")


def test_warmstart_preserves_v5_3_weights():
    """After warm-start, classes 0..6 weights bitwise-equal v5.3's;
    classes 7..9 are zero (in both frag_input_embed rows and frag_head)."""
    print("test_warmstart_preserves_v5_3_weights")
    torch.manual_seed(0)
    m_v53 = build_fragment_stage2(capacity="1M",
                                   num_fragments=V5_3_NUM_FRAGMENTS)
    # Fill v5.3 weights with recognizable patterns to verify transfer
    with torch.no_grad():
        for n, p in m_v53.named_parameters():
            p.fill_(hash(n) % 100 / 100.0 + 0.01)
    v53_state = m_v53.state_dict()

    m_b5 = build_fragment_stage2(capacity="1M",
                                  num_fragments=DEFAULT_NUM_FRAGMENTS)
    status = load_v5_3_warmstart(m_b5, v53_state, strict_shape_check=True,
                                  verbose=False)

    # Status checks
    _assert(status["frag_input_embed.weight"] == "expanded",
            "frag_input_embed should be 'expanded'")
    _assert(status["frag_head.weight"] == "expanded",
            "frag_head.weight should be 'expanded'")
    _assert(status["frag_head.bias"] == "expanded",
            "frag_head.bias should be 'expanded'")
    _assert(status["atom_embed.weight"] == "transferred",
            "atom_embed should be 'transferred'")

    # Weight-content checks
    fh_w_v53 = v53_state["frag_head.weight"]    # (7, d)
    fh_w_b5  = m_b5.frag_head.weight            # (10, d)
    _assert(torch.allclose(fh_w_b5[:7], fh_w_v53),
            "frag_head rows 0..6 not transferred")
    _assert(torch.allclose(fh_w_b5[7:], torch.zeros_like(fh_w_b5[7:])),
            "frag_head rows 7..9 not zero-padded")

    fi_v53 = v53_state["frag_input_embed.weight"]   # (8, d) — 7 classes + MASK
    fi_b5  = m_b5.frag_input_embed.weight            # (11, d)
    _assert(torch.allclose(fi_b5[:7], fi_v53[:7]),
            "frag_input rows 0..6 not transferred")
    _assert(torch.allclose(fi_b5[7:10], torch.zeros_like(fi_b5[7:10])),
            "frag_input rows 7..9 should be zero")
    _assert(torch.allclose(fi_b5[10], fi_v53[7]),
            "frag_input MASK row (B5 row 10) not from v5.3 row 7")

    # atom_embed and other tensors transferred verbatim
    _assert(torch.allclose(m_b5.atom_embed.weight,
                            v53_state["atom_embed.weight"]),
            "atom_embed not transferred verbatim")
    _print_pass("warmstart_preserves_v5_3_weights")


# ─── Driver ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 Batch 5 — terminal_fragment_diffusion tests")
    print("=" * 70)
    test_vocab_consistency_with_smarts_file()
    test_mask_class_idx()
    test_corruption_padding_unchanged()
    test_corruption_alpha_zero()
    test_class_weights_inverse_freq()
    test_param_counts_per_capacity()
    test_unknown_capacity_raises()
    test_forward_shape()
    test_bias_disabled_path()
    test_gradient_flow_full()
    test_loss_decreases()
    test_loss_initial_near_log_K()
    test_padding_zero_loss_contribution()
    test_sample_shape_vocab_padding()
    test_warmstart_tensor_names_match_v5_3()
    test_warmstart_preserves_v5_3_weights()
    print()
    print("All 16 model tests PASSED")


if __name__ == "__main__":
    main()
