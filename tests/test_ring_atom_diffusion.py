"""
HiMoFlow v5.4 — Unit tests for ring_atom_diffusion.py (Batch 4).

Tests cover:
  - Vocab/constants are consistent with ring_layout_decoder.py
  - Diffusion schedule + corruption mechanics
  - Model construction at all capacity presets
  - Forward shapes (with edge bias on and off)
  - Gradient flow through every param
  - Loss decreases on synthetic data
  - Sampling: shape, vocab range, padding correctness, aromatic
    constraint enforcement
  - Aromatic constraint mask construction
  - End-to-end: pipe a real (decoded) layout through A2 and check the
    output is a valid scaffold the decoder accepts

Run with:
    python tests/test_ring_atom_diffusion.py
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np
import torch

# Make package importable
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from meanflow.ring_atom_diffusion import (
    M_MAX, N_ATOM_CLASSES, N_BOND_CLASSES, MASK_ATOM,
    AROMATIC_ATOM_IDS, ALIPHATIC_ATOM_IDS, ATOM_PAD,
    CAPACITY_PRESETS,
    alpha_bar, corrupt_atoms,
    build_ring_atom_diffusion, count_parameters,
)
# Also import the decoder constants to check consistency
from meanflow.ring_layout_decoder import (
    AROMATIC_ATOM_IDS as DECODER_AROMATIC_IDS,
    ATOM_PAD as DECODER_ATOM_PAD,
    M_MAX as DECODER_M_MAX,
    BOND_NONE, BOND_SINGLE, BOND_AROMATIC,
)


def _print_pass(name): print(f"  PASS  {name}")
def _assert(cond, msg):
    if not cond: raise AssertionError(msg)


# ─── Helpers ─────────────────────────────────────────────────────────

def _make_naphthalene_batch(B: int = 4, seed: int = 0):
    """B copies of naphthalene: 10 aromatic c, 11 aromatic bonds.
    Matches what the decoder would produce for R=[1,1,0,0], F[0,1]=FUSED."""
    torch.manual_seed(seed)
    atom_ids = torch.zeros(B, M_MAX, dtype=torch.long)
    atom_ids[:, :10] = 1  # all aromatic c
    atom_mask = torch.zeros(B, M_MAX, dtype=torch.bool)
    atom_mask[:, :10] = True
    arom_mask = atom_mask.clone()
    bond_classes = torch.zeros(B, M_MAX, M_MAX, dtype=torch.long)
    naph_bonds = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,4),(0,9)]
    for (i, j) in naph_bonds:
        bond_classes[:, i, j] = BOND_AROMATIC
        bond_classes[:, j, i] = BOND_AROMATIC
    condition = torch.randn(B, 2)
    return dict(
        atom_ids=atom_ids, bond_classes=bond_classes,
        atom_mask=atom_mask, arom_mask=arom_mask, condition=condition,
    )


# ─── Tests: constants consistency ───────────────────────────────────

def test_constants_match_decoder():
    print("test_constants_match_decoder")
    _assert(M_MAX == DECODER_M_MAX, f"M_MAX mismatch: {M_MAX} vs {DECODER_M_MAX}")
    _assert(ATOM_PAD == DECODER_ATOM_PAD, f"ATOM_PAD mismatch")
    _assert(set(AROMATIC_ATOM_IDS) == DECODER_AROMATIC_IDS,
            f"aromatic IDs differ: {set(AROMATIC_ATOM_IDS)} vs {DECODER_AROMATIC_IDS}")
    # ALIPHATIC + AROMATIC + PAD should partition vocab
    expected = set(range(1, N_ATOM_CLASSES))  # all non-PAD IDs
    actual = set(AROMATIC_ATOM_IDS) | set(ALIPHATIC_ATOM_IDS)
    _assert(expected == actual, f"aromatic+aliphatic don't partition vocab: missing {expected-actual}, extra {actual-expected}")
    _assert(MASK_ATOM == N_ATOM_CLASSES, f"MASK_ATOM should be N_ATOM_CLASSES")
    _assert(N_BOND_CLASSES == 3, "scaffold bonds: none/single/aromatic only")
    _print_pass("constants_match_decoder")


# ─── Tests: noise schedule ──────────────────────────────────────────

def test_alpha_bar_endpoints():
    print("test_alpha_bar_endpoints")
    _assert(alpha_bar(torch.tensor([0.0]), "cosine").item() < 1e-6, "α(0)=0")
    _assert(abs(alpha_bar(torch.tensor([1.0]), "cosine").item() - 1.0) < 1e-6, "α(1)=1")
    _print_pass("alpha_bar_endpoints")


def test_corruption_padding_unchanged():
    """Corruption must NEVER mask padding positions — they stay ATOM_PAD."""
    print("test_corruption_padding_unchanged")
    torch.manual_seed(0)
    B, T = 8, M_MAX
    atom_ids = torch.zeros(B, T, dtype=torch.long)
    atom_ids[:, :5] = 3  # 5 valid atoms
    atom_mask = torch.zeros(B, T, dtype=torch.bool)
    atom_mask[:, :5] = True
    alpha = torch.full((B,), 0.9)  # high mask prob
    atom_ids_t, is_masked = corrupt_atoms(atom_ids, atom_mask, alpha)
    # Padding never changes
    _assert((atom_ids_t[:, 5:] == ATOM_PAD).all(), "padding got corrupted")
    _assert(not is_masked[:, 5:].any(), "is_masked True at padding positions")
    # Valid positions: ~90% should be MASK
    valid_masked = is_masked[:, :5].float().mean()
    _assert(0.7 < valid_masked < 1.0, f"unexpected mask rate {valid_masked:.3f}")
    _print_pass("corruption_padding_unchanged")


def test_corruption_alpha_zero():
    print("test_corruption_alpha_zero")
    atom_ids = torch.tensor([[1, 3, 5, 0, 0]])
    atom_mask = torch.tensor([[True, True, True, False, False]])
    atom_ids_t, is_masked = corrupt_atoms(atom_ids, atom_mask, torch.zeros(1))
    _assert(torch.equal(atom_ids_t, atom_ids), "α=0 should leave all unchanged")
    _assert(not is_masked.any(), "α=0 should mask nothing")
    _print_pass("corruption_alpha_zero")


# ─── Tests: model construction ─────────────────────────────────────

def test_param_counts_per_capacity():
    print("test_param_counts_per_capacity")
    expected = {
        "1M":   (800_000, 1_400_000),
        "3M":   (3_000_000, 4_500_000),
        "10M":  (7_500_000, 11_000_000),
        "30M":  (20_000_000, 30_000_000),
    }
    for cap, (lo, hi) in expected.items():
        m = build_ring_atom_diffusion(capacity=cap)
        n = count_parameters(m)
        _assert(lo <= n <= hi,
                f"{cap} got {n:,} params, expected in [{lo:,}, {hi:,}]")
        print(f"    {cap:>4}: {n:>12,} params")
    _print_pass("param_counts_per_capacity")


def test_unknown_capacity_raises():
    print("test_unknown_capacity_raises")
    try:
        build_ring_atom_diffusion(capacity="99T")
        _assert(False, "should have raised")
    except ValueError as e:
        _assert("Unknown capacity" in str(e), f"wrong msg: {e}")
    _print_pass("unknown_capacity_raises")


# ─── Tests: forward ────────────────────────────────────────────────

def test_forward_shape():
    print("test_forward_shape")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    B = 3
    atom_ids_t = torch.zeros(B, M_MAX, dtype=torch.long)
    bond_classes = torch.zeros(B, M_MAX, M_MAX, dtype=torch.long)
    atom_mask = torch.ones(B, M_MAX, dtype=torch.bool)
    alpha = torch.rand(B)
    cond = torch.randn(B, 2)
    logits = m(atom_ids_t=atom_ids_t, bond_classes=bond_classes,
               atom_mask=atom_mask, alpha=alpha, condition=cond)
    _assert(logits.shape == (B, M_MAX, N_ATOM_CLASSES),
            f"logit shape {logits.shape} != ({B}, {M_MAX}, {N_ATOM_CLASSES})")
    _print_pass("forward_shape")


def test_edge_attn_disabled_path():
    """Building with edge_attn_enabled=False should still work."""
    print("test_edge_attn_disabled_path")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M", edge_attn_enabled=False)
    m.eval()
    batch = _make_naphthalene_batch(B=2)
    out = m.compute_loss(**batch)
    _assert(torch.isfinite(out["loss"]), "loss not finite for ablation path")
    _print_pass("edge_attn_disabled_path")


def test_gradient_flow_full():
    """All params should accumulate gradient after a few steps.

    Subtlety: AdaLN's proj is zero-initialized so blocks start as identity
    (γ=β=0). On step 0 this blocks gradient FROM the loss BACK through
    AdaLN to cond_in_proj — the AdaLN's own params still get gradient
    (their grad depends on their input, not their weight), but params
    UPSTREAM of AdaLN don't see signal until the AdaLN weights become
    nonzero. After one optimizer step, AdaLN.proj.weight is nonzero
    and cond_in_proj begins accumulating gradient normally. So we run
    two steps and check by-then all params have moved from init.
    """
    print("test_gradient_flow_full")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.train()
    m.cfg_drop_prob = 0.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    init_state = {n: p.detach().clone() for n, p in m.named_parameters()}

    batch = _make_naphthalene_batch(B=4)
    for _ in range(2):
        out = m.compute_loss(**batch)
        opt.zero_grad(); out["loss"].backward(); opt.step()

    n_changed = sum(
        1 for n, p in m.named_parameters()
        if (p.detach() - init_state[n]).abs().sum() > 1e-9
    )
    n_total = sum(1 for _ in m.parameters())
    _assert(n_changed == n_total,
            f"only {n_changed}/{n_total} params updated after 2 steps")
    _print_pass("gradient_flow_full")


# ─── Tests: training ──────────────────────────────────────────────

def test_loss_finite_and_decreases():
    print("test_loss_finite_and_decreases")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    batch = _make_naphthalene_batch(B=8)
    losses = []
    for step in range(40):
        out = m.compute_loss(**batch)
        _assert(torch.isfinite(out["loss"]), f"loss NaN at step {step}")
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())
    _assert(losses[-1] < losses[0] * 0.3,
            f"loss didn't decrease: {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"    loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    _print_pass("loss_finite_and_decreases")


def test_loss_initial_near_log_K():
    """Heads have zero-init bias + Kaiming weights → CE should be near
    log(N_ATOM_CLASSES) = log(10) ≈ 2.30 at init."""
    print("test_loss_initial_near_log_K")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.train()
    m.cfg_drop_prob = 0.0
    batch = _make_naphthalene_batch(B=32)
    out = m.compute_loss(**batch)
    expected = math.log(N_ATOM_CLASSES)
    actual = out["loss"].item()
    _assert(abs(actual - expected) < 1.0,
            f"initial loss {actual:.3f} not near log({N_ATOM_CLASSES})={expected:.3f}")
    _print_pass("loss_initial_near_log_K")


def test_loss_padding_zero_contribution():
    """Padding positions must contribute 0 to loss — verify by shrinking
    the valid region and confirming loss only changes if valid atoms change."""
    print("test_loss_padding_zero_contribution")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    batch = _make_naphthalene_batch(B=4)

    # Modify the PADDING region of atom_ids (positions 10-23 are PAD)
    # and confirm loss is identical
    out_a = m.compute_loss(**batch)
    batch_b = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    batch_b["atom_ids"][:, 10:] = 5  # change padding values to non-PAD
    # Note: corruption may flip these but it shouldn't matter — atom_mask=False there.

    # Reset noise to be deterministic
    torch.manual_seed(42)
    out_a = m.compute_loss(**batch)
    torch.manual_seed(42)
    out_b = m.compute_loss(**batch_b)
    _assert(abs(out_a["loss"].item() - out_b["loss"].item()) < 1e-5,
            f"loss differs ({out_a['loss'].item()} vs {out_b['loss'].item()}); "
            f"padding contributed to loss")
    _print_pass("loss_padding_zero_contribution")


# ─── Tests: sampling ──────────────────────────────────────────────

def test_sample_shape_and_vocab():
    print("test_sample_shape_and_vocab")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    batch = _make_naphthalene_batch(B=4)
    samples = m.sample(
        bond_classes=batch["bond_classes"], atom_mask=batch["atom_mask"],
        arom_mask=batch["arom_mask"], condition=batch["condition"],
        n_steps=8, seed=42,
    )
    _assert(samples.shape == (4, M_MAX), f"shape {samples.shape}")
    _assert((samples >= 0).all() and (samples < N_ATOM_CLASSES).all(),
            "atom IDs out of vocab range")
    _assert((samples != MASK_ATOM).all(), "MASK_ATOM appears in output")
    _print_pass("sample_shape_and_vocab")


def test_sample_padding_is_pad():
    """Sampler output at atom_mask=False positions must be ATOM_PAD."""
    print("test_sample_padding_is_pad")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    batch = _make_naphthalene_batch(B=4)
    samples = m.sample(
        bond_classes=batch["bond_classes"], atom_mask=batch["atom_mask"],
        arom_mask=batch["arom_mask"], condition=batch["condition"],
        n_steps=8, seed=42,
    )
    _assert((samples[~batch["atom_mask"]] == ATOM_PAD).all(),
            "padding position predicted non-PAD class")
    _print_pass("sample_padding_is_pad")


def test_sample_aromatic_constraint_satisfied():
    """At every aromatic-required position, predicted ID must be in
    AROMATIC_ATOM_IDS. This is the key safety property."""
    print("test_sample_aromatic_constraint_satisfied")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    batch = _make_naphthalene_batch(B=8)
    samples = m.sample(
        bond_classes=batch["bond_classes"], atom_mask=batch["atom_mask"],
        arom_mask=batch["arom_mask"], condition=batch["condition"],
        n_steps=8, seed=42,
    )
    arom_set = torch.tensor(AROMATIC_ATOM_IDS)
    pred_in_arom = (samples.unsqueeze(-1) == arom_set).any(dim=-1)
    # At valid + aromatic-required positions, pred MUST be aromatic
    valid_arom = batch["atom_mask"] & batch["arom_mask"]
    _assert(pred_in_arom[valid_arom].all(),
            "aromatic constraint violated at sample time")
    _print_pass("sample_aromatic_constraint_satisfied")


def test_sample_aliphatic_position_can_be_aliphatic():
    """At aliphatic-required positions, the sampler should be able to
    pick aliphatic IDs (no false constraint forbidding them)."""
    print("test_sample_aliphatic_position_can_be_aliphatic")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    # Build a 6-aliphatic-ring batch (cyclohexane: 6 single-bonded C)
    B = 16
    atom_ids = torch.zeros(B, M_MAX, dtype=torch.long)
    atom_ids[:, :6] = 3  # C
    atom_mask = torch.zeros(B, M_MAX, dtype=torch.bool); atom_mask[:, :6] = True
    arom_mask = torch.zeros(B, M_MAX, dtype=torch.bool)  # all aliphatic
    bond_classes = torch.zeros(B, M_MAX, M_MAX, dtype=torch.long)
    for i in range(6):
        j = (i + 1) % 6
        bond_classes[:, i, j] = BOND_SINGLE; bond_classes[:, j, i] = BOND_SINGLE
    cond = torch.randn(B, 2)

    samples = m.sample(
        bond_classes=bond_classes, atom_mask=atom_mask, arom_mask=arom_mask,
        condition=cond, n_steps=8, seed=42,
    )
    # At valid + aliphatic positions, samples should mostly be from
    # ALIPHATIC_ATOM_IDS (untrained — but the constraint allows them).
    aliph_set = torch.tensor(ALIPHATIC_ATOM_IDS)
    pred_in_aliph = (samples.unsqueeze(-1) == aliph_set).any(dim=-1)
    valid_aliph = atom_mask & (~arom_mask)
    # Untrained model: at least SOME samples should be aliphatic (constraint
    # doesn't forbid them; only forbids ATOM_PAD).
    _assert(pred_in_aliph[valid_aliph].any(),
            "no aliphatic predictions at aliphatic positions — constraint over-restricts")
    # All predictions should be in non-PAD valid range
    _assert((samples[valid_aliph] != ATOM_PAD).all(),
            "ATOM_PAD predicted at valid aliphatic position")
    _print_pass("sample_aliphatic_position_can_be_aliphatic")


# ─── Tests: forbidden mask construction ──────────────────────────

def test_forbidden_mask_padding():
    print("test_forbidden_mask_padding")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M")
    atom_mask = torch.tensor([[True, True, False, False]])
    arom_mask = torch.tensor([[True, False, False, False]])
    forbid = m._build_class_forbidden_mask(atom_mask, arom_mask)
    # Position 2,3 are padding: only ATOM_PAD allowed
    _assert((~forbid[0, 2, ATOM_PAD]).item(), "PAD class forbidden at padding")
    for c in range(1, N_ATOM_CLASSES):
        _assert(forbid[0, 2, c].item(), f"class {c} should be forbidden at padding")
    _print_pass("forbidden_mask_padding")


def test_forbidden_mask_aromatic():
    print("test_forbidden_mask_aromatic")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M")
    atom_mask = torch.tensor([[True]])
    arom_mask = torch.tensor([[True]])
    forbid = m._build_class_forbidden_mask(atom_mask, arom_mask)
    # ATOM_PAD forbidden; aliphatic IDs forbidden; aromatic IDs allowed
    _assert(forbid[0, 0, ATOM_PAD].item(), "ATOM_PAD should be forbidden at aromatic position")
    for aliph in ALIPHATIC_ATOM_IDS:
        _assert(forbid[0, 0, aliph].item(),
                f"aliphatic ID {aliph} should be forbidden at aromatic position")
    for arom in AROMATIC_ATOM_IDS:
        _assert((~forbid[0, 0, arom]).item(),
                f"aromatic ID {arom} should be allowed at aromatic position")
    _print_pass("forbidden_mask_aromatic")


def test_forbidden_mask_aliphatic():
    print("test_forbidden_mask_aliphatic")
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M")
    atom_mask = torch.tensor([[True]])
    arom_mask = torch.tensor([[False]])
    forbid = m._build_class_forbidden_mask(atom_mask, arom_mask)
    # ATOM_PAD forbidden; everything else allowed
    _assert(forbid[0, 0, ATOM_PAD].item(),
            "ATOM_PAD should be forbidden at aliphatic position")
    for c in range(1, N_ATOM_CLASSES):
        _assert((~forbid[0, 0, c]).item(),
                f"class {c} should be allowed at aliphatic position")
    _print_pass("forbidden_mask_aliphatic")


# ─── End-to-end test: A1 → decoder → A2 → re-decode ──────────────

def test_end_to_end_with_real_layout():
    """Take a known good layout, decode it to (bond_classes, atom_mask,
    arom_mask), feed through A2, and verify the resulting atom_ids form
    a layout the decoder accepts (round-trip)."""
    print("test_end_to_end_with_real_layout")
    from meanflow.ring_layout_decoder import (
        compute_atom_count, aromatic_constraint_mask, build_bond_classes,
        decode_layout_to_scaffold,
    )
    # Naphthalene layout
    R = np.array([1, 1, 0, 0])
    F_mat = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
    L_mat = np.zeros((4, 4), dtype=np.int64)
    P_len = np.zeros((4, 4), dtype=np.int64)
    P_pos = np.zeros((4, 4), dtype=np.int64)
    M_total = compute_atom_count(R, F_mat, L_mat, P_len, P_pos)

    # Build (bond_classes, atom_mask, arom_mask) batch tensors
    bond_classes_np, atom_mask_np = build_bond_classes(
        R, F_mat, L_mat, P_len, P_pos, M_MAX_out=M_MAX,
    )
    arom_mask_np = aromatic_constraint_mask(R, F_mat, L_mat, P_len, P_pos, M_MAX_out=M_MAX)

    # Replicate as a tiny batch
    B = 4
    bond_classes = torch.from_numpy(bond_classes_np).long().unsqueeze(0).expand(B, -1, -1).clone()
    atom_mask = torch.from_numpy(atom_mask_np).bool().unsqueeze(0).expand(B, -1).clone()
    arom_mask = torch.from_numpy(arom_mask_np).bool().unsqueeze(0).expand(B, -1).clone()
    cond = torch.randn(B, 2)

    # Sample atoms with an UNTRAINED A2 — should still produce a valid
    # scaffold because the aromatic constraint is enforced in the sampler.
    torch.manual_seed(0)
    m = build_ring_atom_diffusion(capacity="1M"); m.eval()
    samples = m.sample(
        bond_classes=bond_classes, atom_mask=atom_mask, arom_mask=arom_mask,
        condition=cond, n_steps=8, seed=42,
    )

    # Re-decode with the predicted atom_ids: this should succeed because
    # (a) bonds match the layout, (b) aromatic constraint is satisfied
    # for all aromatic-required atoms.
    n_passed = 0
    for b in range(B):
        atom_ids_compact = samples[b][:M_total].cpu().numpy()
        try:
            decode_layout_to_scaffold(R, F_mat, L_mat, P_len, P_pos, atom_ids_compact)
            n_passed += 1
        except Exception as e:
            print(f"    sample {b} failed decode: {type(e).__name__}: {e}")
    _assert(n_passed == B,
            f"only {n_passed}/{B} round-tripped through decoder")
    _print_pass("end_to_end_with_real_layout")


# ─── Driver ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 Batch 4 — ring_atom_diffusion tests")
    print("=" * 70)
    test_constants_match_decoder()
    test_alpha_bar_endpoints()
    test_corruption_padding_unchanged()
    test_corruption_alpha_zero()
    test_param_counts_per_capacity()
    test_unknown_capacity_raises()
    test_forward_shape()
    test_edge_attn_disabled_path()
    test_gradient_flow_full()
    test_loss_finite_and_decreases()
    test_loss_initial_near_log_K()
    test_loss_padding_zero_contribution()
    test_sample_shape_and_vocab()
    test_sample_padding_is_pad()
    test_sample_aromatic_constraint_satisfied()
    test_sample_aliphatic_position_can_be_aliphatic()
    test_forbidden_mask_padding()
    test_forbidden_mask_aromatic()
    test_forbidden_mask_aliphatic()
    test_end_to_end_with_real_layout()
    print()
    print("All 20 tests PASSED")


if __name__ == "__main__":
    main()
