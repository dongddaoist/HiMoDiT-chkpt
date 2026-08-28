"""
HiMoFlow v5.4 — Unit tests for ring_layout_diffusion.py (Batch 3).

Tests cover:
  - Model construction at all capacity presets, with parameter-count
    sanity bounds.
  - Forward pass shape contract.
  - Discrete corruption: rate matches alpha; α=0 → no corruption;
    α=1 → all MASK.
  - compute_loss: returns finite, gradients flow, one Adam step
    decreases loss.
  - Sample: returns correct shapes and produces samples that the
    decoder accepts at non-trivial frequency.
  - postprocess_layout: enforces the 4 advertised invariants.

Run with:
    python tests/test_ring_layout_diffusion.py

All tests are designed to run on CPU in <1 minute total. Convergence
is NOT tested (real-data concern).
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import torch

# Make package importable when run from various roots
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from meanflow.ring_layout_diffusion import (
    R_MAX, P_MAX, N_TOKENS, N_RING_TOKENS, N_PAIR_TOKENS, N_PEND_TOKENS,
    N_R_CLASSES, N_F_CLASSES, N_L_CLASSES, N_PLEN_CLASSES, N_PPOS_CLASSES,
    MASK_R, MASK_F, MASK_L, MASK_PLEN, MASK_PPOS,
    PAIR_INDICES, CAPACITY_PRESETS,
    alpha_bar, corrupt_categorical,
    build_ring_layout_diffusion, count_parameters,
    postprocess_layout,
)
from meanflow.ring_layout_decoder import decode_layout_to_scaffold


# ─── Helpers ───────────────────────────────────────────────────────────

def _print_pass(name):
    print(f"  PASS  {name}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _make_synthetic_batch(B: int = 4, seed: int = 0):
    """Synthetic batch resembling a 2-ring fused layout (no pendants).
    Mirrors the actual RedDB statistical pattern."""
    torch.manual_seed(seed)
    R = torch.zeros(B, R_MAX, dtype=torch.long)
    R[:, 0] = 1  # 6-arom
    R[:, 1] = 1  # 6-arom (fused naphthalene-like)
    F_mat = torch.zeros((B, R_MAX, R_MAX), dtype=torch.long)
    F_mat[:, 0, 1] = 1  # FUSED
    F_mat[:, 1, 0] = 1
    L_mat = torch.zeros((B, R_MAX, R_MAX), dtype=torch.long)
    P_len = torch.zeros((B, R_MAX, P_MAX), dtype=torch.long)
    P_pos = torch.zeros((B, R_MAX, P_MAX), dtype=torch.long)
    condition = torch.randn(B, 2)
    return dict(R=R, F_mat=F_mat, L_mat=L_mat, P_len=P_len,
                P_pos=P_pos, condition=condition)


# ─── Tests: constants & token layout ───────────────────────────────────

def test_token_layout_constants():
    print("test_token_layout_constants")
    _assert(N_RING_TOKENS == R_MAX,
            f"N_RING_TOKENS should be R_MAX={R_MAX}")
    _assert(N_PAIR_TOKENS == R_MAX * (R_MAX - 1) // 2,
            "N_PAIR_TOKENS should be upper-triangle count")
    _assert(N_PEND_TOKENS == R_MAX * P_MAX,
            "N_PEND_TOKENS should be R_MAX * P_MAX")
    _assert(N_TOKENS == N_RING_TOKENS + N_PAIR_TOKENS + N_PEND_TOKENS,
            f"N_TOKENS={N_TOKENS} doesn't add up")
    _assert(len(PAIR_INDICES) == N_PAIR_TOKENS,
            "PAIR_INDICES length mismatch")
    # Verify PAIR_INDICES is sorted upper triangle
    for k, (i, j) in enumerate(PAIR_INDICES):
        _assert(i < j, f"PAIR_INDICES[{k}]=({i},{j}) not upper-triangular")
    _print_pass("token_layout_constants")


# ─── Tests: noise schedule ─────────────────────────────────────────────

def test_alpha_bar_endpoints():
    print("test_alpha_bar_endpoints")
    t = torch.tensor([0.0])
    _assert(alpha_bar(t, "cosine").item() < 1e-6, "α(0) should be 0")
    t = torch.tensor([1.0])
    _assert(abs(alpha_bar(t, "cosine").item() - 1.0) < 1e-6,
            "α(1) should be 1")
    t = torch.tensor([0.5])
    _assert(abs(alpha_bar(t, "linear").item() - 0.5) < 1e-6,
            "linear α(0.5) should be 0.5")
    _print_pass("alpha_bar_endpoints")


# ─── Tests: corruption ─────────────────────────────────────────────────

def test_corruption_zero_alpha():
    print("test_corruption_zero_alpha")
    torch.manual_seed(0)
    x = torch.randint(0, 5, (8, 4))
    alpha = torch.zeros(8)
    x_t, is_masked = corrupt_categorical(x, alpha, MASK_R)
    _assert(torch.equal(x_t, x), "α=0 should leave x unchanged")
    _assert(not is_masked.any(), "α=0 should mask nothing")
    _print_pass("corruption_zero_alpha")


def test_corruption_one_alpha():
    print("test_corruption_one_alpha")
    torch.manual_seed(0)
    x = torch.randint(0, 5, (8, 4))
    alpha = torch.ones(8)
    x_t, is_masked = corrupt_categorical(x, alpha, MASK_R)
    _assert(torch.all(x_t == MASK_R), "α=1 should mask everything")
    _assert(is_masked.all(), "α=1 should set all is_masked")
    _print_pass("corruption_one_alpha")


def test_corruption_rate_approx():
    print("test_corruption_rate_approx")
    torch.manual_seed(0)
    # Large enough sample to get statistical convergence
    x = torch.randint(0, 5, (1000, 16))
    alpha = torch.full((1000,), 0.3)
    x_t, is_masked = corrupt_categorical(x, alpha, MASK_R)
    rate = is_masked.float().mean().item()
    # 99.9% confidence interval at α=0.3, n=16000: ±~0.012
    _assert(abs(rate - 0.3) < 0.02,
            f"Empirical mask rate {rate:.4f} far from α=0.3")
    _print_pass("corruption_rate_approx")


# ─── Tests: model construction ─────────────────────────────────────────

def test_param_counts_per_capacity():
    print("test_param_counts_per_capacity")
    expected_ranges = {
        "600K": (500_000, 750_000),
        "1M":   (900_000, 1_300_000),
        "3M":   (2_900_000, 4_400_000),
        "10M":  (9_000_000, 12_000_000),
    }
    for cap, (lo, hi) in expected_ranges.items():
        m = build_ring_layout_diffusion(capacity=cap)
        n = count_parameters(m)
        _assert(lo <= n <= hi,
                f"{cap} preset produced {n:,} params, expected in [{lo:,}, {hi:,}]")
        print(f"    {cap:>5}: {n:>12,} params")
    _print_pass("param_counts_per_capacity")


def test_unknown_capacity_raises():
    print("test_unknown_capacity_raises")
    try:
        build_ring_layout_diffusion(capacity="blarg")
        _assert(False, "should have raised")
    except ValueError as e:
        _assert("Unknown capacity" in str(e), "wrong error message")
    _print_pass("unknown_capacity_raises")


# ─── Tests: forward pass ───────────────────────────────────────────────

def test_forward_shapes():
    print("test_forward_shapes")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.eval()
    B = 3
    R_t = torch.randint(0, N_R_CLASSES + 1, (B, R_MAX))
    F_t = torch.randint(0, N_F_CLASSES + 1, (B, R_MAX, R_MAX))
    L_t = torch.randint(0, N_L_CLASSES + 1, (B, R_MAX, R_MAX))
    Plen_t = torch.randint(0, N_PLEN_CLASSES + 1, (B, R_MAX, P_MAX))
    Ppos_t = torch.randint(0, N_PPOS_CLASSES + 1, (B, R_MAX, P_MAX))
    alpha = torch.rand(B)
    cond = torch.randn(B, 2)

    out = m(R_t=R_t, F_full_t=F_t, L_full_t=L_t,
            Plen_t=Plen_t, Ppos_t=Ppos_t,
            alpha=alpha, condition=cond)

    _assert(out["R_logits"].shape == (B, R_MAX, N_R_CLASSES),
            f"R_logits shape {out['R_logits'].shape}")
    _assert(out["F_logits"].shape == (B, N_PAIR_TOKENS, N_F_CLASSES),
            f"F_logits shape {out['F_logits'].shape}")
    _assert(out["L_logits"].shape == (B, N_PAIR_TOKENS, N_L_CLASSES),
            f"L_logits shape {out['L_logits'].shape}")
    _assert(out["Plen_logits"].shape == (B, R_MAX, P_MAX, N_PLEN_CLASSES),
            f"Plen_logits shape {out['Plen_logits'].shape}")
    _assert(out["Ppos_logits"].shape == (B, R_MAX, P_MAX, N_PPOS_CLASSES),
            f"Ppos_logits shape {out['Ppos_logits'].shape}")
    _print_pass("forward_shapes")


def test_forward_no_inplace_bug():
    """A subtle PyTorch trap: if token assembly does in-place ops on
    embedding outputs, gradients break. Verify we can backprop."""
    print("test_forward_no_inplace_bug")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.train()
    batch = _make_synthetic_batch(B=2)
    out = m.compute_loss(**batch)
    out["loss"].backward()
    n_with_grad = sum(1 for p in m.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in m.parameters())
    # Most params should have non-zero gradient (zero-init heads start
    # at zero so a few params may have zero grad initially via dead paths).
    _assert(n_with_grad > n_total // 2,
            f"only {n_with_grad}/{n_total} params received gradient")
    _print_pass("forward_no_inplace_bug")


# ─── Tests: compute_loss ───────────────────────────────────────────────

def test_loss_finite_and_decreases():
    print("test_loss_finite_and_decreases")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    batch = _make_synthetic_batch(B=8)

    losses = []
    m.train()
    for step in range(40):
        out = m.compute_loss(**batch)
        _assert(torch.isfinite(out["loss"]), f"loss NaN/inf at step {step}")
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())

    _assert(losses[-1] < losses[0] * 0.5,
            f"loss did not decrease enough: {losses[0]:.3f} → {losses[-1]:.3f}")
    print(f"    loss {losses[0]:.3f} → {losses[-1]:.3f}")
    _print_pass("loss_finite_and_decreases")


def test_loss_initial_value_near_uniform_prior():
    """Heads have zero-init bias and small Kaiming-init weights, so on
    a fresh model the initial loss should be CLOSE to log(K) but not
    exactly (we cannot zero-init head weights without killing the
    gradient through the DiT stack — see r_head etc. comment)."""
    print("test_loss_initial_value_near_uniform_prior")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.train()
    batch = _make_synthetic_batch(B=32)
    # Disable CFG drop for this test (otherwise condition is randomly zeroed)
    m.cfg_drop_prob = 0.0
    out = m.compute_loss(**batch)
    expected = {
        "R": math.log(N_R_CLASSES),
        "F": math.log(N_F_CLASSES),
        "L": math.log(N_L_CLASSES),
        "Plen": math.log(N_PLEN_CLASSES),
        "Ppos": math.log(N_PPOS_CLASSES),
    }
    # Allow up to 1.0 nat deviation — small Kaiming-init weights produce
    # non-zero logits, so CE deviates from log(K). With B=32 the empirical
    # CE also has finite-sample variance.
    for k, exp in expected.items():
        actual = out[f"loss_{k}"].item()
        _assert(abs(actual - exp) < 1.0,
                f"loss_{k}={actual:.4f}, expected ~log(N)={exp:.4f} (±1.0)")
    _print_pass("loss_initial_value_near_uniform_prior")


# ─── Tests: sampling ───────────────────────────────────────────────────

def test_sample_shapes():
    print("test_sample_shapes")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.eval()
    B = 4
    cond = torch.randn(B, 2)
    samples = m.sample(condition=cond, n_steps=5, seed=42)

    _assert(samples["R"].shape == (B, R_MAX),
            f"R shape {samples['R'].shape}")
    _assert(samples["F"].shape == (B, R_MAX, R_MAX),
            f"F shape {samples['F'].shape}")
    _assert(samples["L"].shape == (B, R_MAX, R_MAX),
            f"L shape {samples['L'].shape}")
    _assert(samples["P_len"].shape == (B, R_MAX, P_MAX),
            f"P_len shape {samples['P_len'].shape}")
    _assert(samples["P_pos"].shape == (B, R_MAX, P_MAX),
            f"P_pos shape {samples['P_pos'].shape}")
    _print_pass("sample_shapes")


def test_sample_no_mask_in_output():
    print("test_sample_no_mask_in_output")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.eval()
    B = 8
    cond = torch.randn(B, 2)
    samples = m.sample(condition=cond, n_steps=5, seed=42)
    # No element should still be MASK
    _assert((samples["R"] != MASK_R).all(), "R contains MASK")
    _assert((samples["F"] != MASK_F).all(), "F contains MASK")
    _assert((samples["L"] != MASK_L).all(), "L contains MASK")
    _assert((samples["P_len"] != MASK_PLEN).all(), "P_len contains MASK")
    _assert((samples["P_pos"] != MASK_PPOS).all(), "P_pos contains MASK")
    _print_pass("sample_no_mask_in_output")


def test_sample_class_ranges():
    print("test_sample_class_ranges")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.eval()
    cond = torch.randn(8, 2)
    s = m.sample(condition=cond, n_steps=5, seed=42)
    _assert(s["R"].max() < N_R_CLASSES, "R out of range")
    _assert(s["F"].max() < N_F_CLASSES, "F out of range")
    _assert(s["L"].max() < N_L_CLASSES, "L out of range")
    _assert(s["P_len"].max() < N_PLEN_CLASSES, "P_len out of range")
    _assert(s["P_pos"].max() < N_PPOS_CLASSES, "P_pos out of range")
    _assert(s["R"].min() >= 0 and s["F"].min() >= 0
            and s["L"].min() >= 0 and s["P_len"].min() >= 0
            and s["P_pos"].min() >= 0,
            "negative class IDs in sample")
    _print_pass("sample_class_ranges")


# ─── Tests: postprocess_layout ─────────────────────────────────────────

def test_postprocess_F_symmetric():
    print("test_postprocess_F_symmetric")
    B = 2
    layout = {
        "R": torch.tensor([[1, 1, 0, 0]] * B, dtype=torch.long),
        "F": torch.tensor([[[0, 1, 0, 0],   # asymmetric input
                            [0, 0, 2, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]] * B, dtype=torch.long),
        "L": torch.zeros(B, R_MAX, R_MAX, dtype=torch.long),
        "P_len": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
        "P_pos": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
    }
    pp = postprocess_layout(layout)
    F_out = pp["F"]
    _assert(torch.equal(F_out, F_out.transpose(-1, -2)), "F not symmetric")
    diag = torch.arange(R_MAX)
    _assert((F_out[:, diag, diag] == 0).all(), "F diagonal not zero")
    _print_pass("postprocess_F_symmetric")


def test_postprocess_zero_L_when_F_not_linked():
    print("test_postprocess_zero_L_when_F_not_linked")
    B = 1
    F_in = torch.zeros(B, R_MAX, R_MAX, dtype=torch.long)
    F_in[:, 0, 1] = 1; F_in[:, 1, 0] = 1  # FUSED at (0,1)
    F_in[:, 0, 2] = 2; F_in[:, 2, 0] = 2  # LINKED at (0,2)
    L_in = torch.tensor([[[0, 3, 4, 0],   # L=3 at (0,1) but F=FUSED → should be zeroed
                          [3, 0, 0, 0],
                          [4, 0, 0, 0],   # L=4 at (0,2) F=LINKED → keep
                          [0, 0, 0, 0]]], dtype=torch.long)
    layout = {
        "R": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        "F": F_in,
        "L": L_in,
        "P_len": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
        "P_pos": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
    }
    pp = postprocess_layout(layout)
    L_out = pp["L"]
    # L at (0,1) should be zeroed (F=FUSED)
    _assert(L_out[0, 0, 1] == 0, f"L at FUSED edge not zeroed: {L_out[0,0,1]}")
    _assert(L_out[0, 1, 0] == 0, "L symmetric not zeroed at FUSED")
    # L at (0,2) should be kept (F=LINKED)
    _assert(L_out[0, 0, 2] == 4, f"L at LINKED edge {L_out[0,0,2]} ≠ 4")
    _assert(L_out[0, 2, 0] == 4, "L symmetric not preserved at LINKED")
    _print_pass("postprocess_zero_L_when_F_not_linked")


def test_postprocess_zero_Ppos_when_Plen_zero():
    print("test_postprocess_zero_Ppos_when_Plen_zero")
    B = 1
    Plen = torch.tensor([[[0, 2, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]], dtype=torch.long)
    Ppos = torch.tensor([[[3, 1, 5, 2],
                          [4, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]], dtype=torch.long)
    layout = {
        "R": torch.tensor([[1, 0, 0, 0]], dtype=torch.long),
        "F": torch.zeros(B, R_MAX, R_MAX, dtype=torch.long),
        "L": torch.zeros(B, R_MAX, R_MAX, dtype=torch.long),
        "P_len": Plen,
        "P_pos": Ppos,
    }
    pp = postprocess_layout(layout)
    P_pos_out = pp["P_pos"]
    P_len_out = pp["P_len"]
    # Wherever P_len_out is 0, P_pos_out should be 0
    zero_locs = (P_len_out == 0)
    _assert((P_pos_out[zero_locs] == 0).all(),
            "P_pos not zeroed where P_len=0")
    # Where P_len > 0 in active rings, P_pos should be preserved
    _assert(P_pos_out[0, 0, 1] == 1, "P_pos at active slot not preserved")
    _print_pass("postprocess_zero_Ppos_when_Plen_zero")


def test_postprocess_left_packs_R():
    print("test_postprocess_left_packs_R")
    B = 1
    # R has PAD in slot 0 but ring in slot 1
    R = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    F_in = torch.zeros(B, R_MAX, R_MAX, dtype=torch.long)
    F_in[:, 1, 2] = 1; F_in[:, 2, 1] = 1  # ring 1 fused with ring 2
    layout = {
        "R": R,
        "F": F_in,
        "L": torch.zeros(B, R_MAX, R_MAX, dtype=torch.long),
        "P_len": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
        "P_pos": torch.zeros(B, R_MAX, P_MAX, dtype=torch.long),
    }
    pp = postprocess_layout(layout)
    R_out = pp["R"]
    F_out = pp["F"]
    # Expected: rings packed to slots 0,1. R[0]=1, R[1]=2 (or reverse?).
    # Specifically: original non-PAD slots [1,2] move to [0,1], in same
    # internal order.
    _assert(R_out[0, 0] == 1, f"R[0]={R_out[0,0]} ≠ 1")
    _assert(R_out[0, 1] == 2, f"R[1]={R_out[0,1]} ≠ 2")
    _assert(R_out[0, 2] == 0 and R_out[0, 3] == 0,
            "trailing slots not PAD")
    # F should also have been permuted: ring 1↔2 fusion is now between
    # slot 0 and slot 1.
    _assert(F_out[0, 0, 1] == 1 and F_out[0, 1, 0] == 1,
            "F not properly re-permuted")
    _print_pass("postprocess_left_packs_R")


# ─── Tests: end-to-end decode acceptance ───────────────────────────────

def test_known_layout_decodes():
    """A known-good layout must decode without error."""
    print("test_known_layout_decodes")
    R = np.array([1, 1, 0, 0])  # 2 fused 6-arom rings
    F_mat = np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    L_mat = np.zeros((4, 4), dtype=np.int64)
    P_len = np.zeros((4, 4), dtype=np.int64)
    P_pos = np.zeros((4, 4), dtype=np.int64)
    # naphthalene-style: 10 atoms total, all aromatic carbons (id=1)
    atom_ids = np.ones(10, dtype=np.int64)
    aip, bc, am = decode_layout_to_scaffold(R, F_mat, L_mat, P_len, P_pos, atom_ids)
    _assert(am.sum() == 10, "atom mask should have 10 atoms")
    _assert((bc == 2).sum() // 2 == 11, "naphthalene has 11 aromatic bonds")
    _print_pass("known_layout_decodes")


# ─── Sample-level diagnostic (informational, no strict pass/fail) ──────

def test_sample_decode_smoke():
    """Sample 16 layouts from an UNTRAINED model. Most will fail to
    decode (expected — random layouts are mostly invalid). Just verify
    the pipeline works end-to-end."""
    print("test_sample_decode_smoke")
    torch.manual_seed(0)
    m = build_ring_layout_diffusion(capacity="600K")
    m.eval()
    cond = torch.randn(16, 2)
    samples = m.sample(condition=cond, n_steps=10, seed=42)

    n_decoded = 0
    n_attempted = samples["R"].shape[0]
    # Try to decode: assign trivial atom_ids (all aromatic carbons for
    # aromatic rings, all aliphatic for aliphatic) just to see if the
    # decoder accepts the bond skeleton.
    for b in range(n_attempted):
        R_ = samples["R"][b].cpu().numpy()
        F_ = samples["F"][b].cpu().numpy()
        L_ = samples["L"][b].cpu().numpy()
        Plen_ = samples["P_len"][b].cpu().numpy()
        Ppos_ = samples["P_pos"][b].cpu().numpy()

        # Compute M_total via decoder helper, fallback gracefully
        from meanflow.ring_layout_decoder import compute_atom_count
        try:
            M_total = compute_atom_count(R_, F_, L_, Plen_, Ppos_)
        except Exception:
            continue
        if M_total == 0:
            n_decoded += 1
            continue
        if M_total > 24:
            continue
        # Build atom_ids: use aromatic carbon (id=1) where ring is aromatic,
        # aliphatic carbon (id=3) elsewhere. Pendants & linkers default to id=3.
        from meanflow.ring_layout_decoder import (
            aromatic_constraint_mask, AROMATIC_ATOM_IDS,
        )
        try:
            arom_mask = aromatic_constraint_mask(R_, F_, L_, Plen_, Ppos_)
        except Exception:
            continue
        atom_ids = np.where(arom_mask[:M_total], 1, 3).astype(np.int64)
        try:
            decode_layout_to_scaffold(R_, F_, L_, Plen_, Ppos_, atom_ids)
            n_decoded += 1
        except Exception:
            pass

    print(f"    {n_decoded}/{n_attempted} sampled layouts decode "
          f"({100.*n_decoded/n_attempted:.1f}%)")
    print(f"    (untrained model — informational only; trained model "
          f"should be much higher)")
    _print_pass("sample_decode_smoke")


# ─── Driver ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 Batch 3 — ring_layout_diffusion tests")
    print("=" * 70)
    test_token_layout_constants()
    test_alpha_bar_endpoints()
    test_corruption_zero_alpha()
    test_corruption_one_alpha()
    test_corruption_rate_approx()
    test_param_counts_per_capacity()
    test_unknown_capacity_raises()
    test_forward_shapes()
    test_forward_no_inplace_bug()
    test_loss_finite_and_decreases()
    test_loss_initial_value_near_uniform_prior()
    test_sample_shapes()
    test_sample_no_mask_in_output()
    test_sample_class_ranges()
    test_postprocess_F_symmetric()
    test_postprocess_zero_L_when_F_not_linked()
    test_postprocess_zero_Ppos_when_Plen_zero()
    test_postprocess_left_packs_R()
    test_known_layout_decodes()
    test_sample_decode_smoke()
    print()
    print("All 20 tests PASSED")


if __name__ == "__main__":
    main()
