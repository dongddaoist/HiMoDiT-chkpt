"""
HiMoFlow v5.4 — tests for ring_layout_decoder (Batch 1.1, with P_pos).

Carries forward all 17 tests from Batch 1, plus new tests verifying:
  - Pendant attaches at the explicit P_pos value
  - Pendant collision with structural (fusion/linker) positions raises
  - Two pendants on same ring sharing a position raises
  - P_pos out of range raises
  - P_pos != 0 with P_len == 0 raises (PAD invariant)
  - list_valid_pendant_positions helper returns correct positions

Run:  cd v5_4_batch1_1 && PYTHONPATH=. python tests/test_ring_layout_decoder.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np

from meanflow.ring_layout_decoder import (
    decode_layout_to_scaffold,
    compute_atom_count,
    build_atom_layout,
    build_bond_classes,
    aromatic_constraint_mask,
    list_valid_pendant_positions,
    R_MAX, M_MAX, P_MAX, L_MAX, P_LEN_MAX,
    RING_PAD, RING_6_AROM, RING_6_ALIPH, RING_5_AROM, RING_5_ALIPH,
    BOND_NONE, BOND_SINGLE, BOND_AROMATIC,
    F_NONE, F_FUSED, F_LINKED,
)


def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")


def _R_pad(*types) -> np.ndarray:
    r = list(types) + [RING_PAD] * (R_MAX - len(types))
    return np.array(r, dtype=np.int64)


def _zero_F() -> np.ndarray:
    return np.zeros((R_MAX, R_MAX), dtype=np.int64)


def _zero_L() -> np.ndarray:
    return np.zeros((R_MAX, R_MAX), dtype=np.int64)


def _zero_P_len() -> np.ndarray:
    return np.zeros((R_MAX, P_MAX), dtype=np.int64)


def _zero_P_pos() -> np.ndarray:
    return np.zeros((R_MAX, P_MAX), dtype=np.int64)


def _F_with(*pairs, value=F_FUSED) -> np.ndarray:
    F = _zero_F()
    for (i, j) in pairs:
        F[i, j] = value
        F[j, i] = value
    return F


def _L_with(*entries) -> np.ndarray:
    L = _zero_L()
    for (i, j, length) in entries:
        L[i, j] = length
        L[j, i] = length
    return L


def _P_len_with(*entries) -> np.ndarray:
    P_len = _zero_P_len()
    for (i, slot, length) in entries:
        P_len[i, slot] = length
    return P_len


def _P_pos_with(*entries) -> np.ndarray:
    P_pos = _zero_P_pos()
    for (i, slot, position) in entries:
        P_pos[i, slot] = position
    return P_pos


# ════════════════════════════════════════════════════════════════════
# Carry-forward (no pendant): rings, fusions, linkers
# ════════════════════════════════════════════════════════════════════

def test_benzene():
    print("test_benzene ...")
    R = _R_pad(RING_6_AROM)
    M = compute_atom_count(R, _zero_F(), _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(M, 6, "benzene atoms")
    bc, _ = build_bond_classes(R, _zero_F(), _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 6, "benzene aromatic bonds")
    print("  PASS")


def test_naphthalene():
    print("test_naphthalene ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    M = compute_atom_count(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(M, 10, "naphthalene atoms")
    bc, _ = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 11, "naphthalene aromatic bonds")
    degrees = (bc > 0).sum(axis=1)
    assert_eq(int((degrees[:10] == 3).sum()), 2, "naphthalene 2 deg-3")
    print("  PASS")


def test_anthracene_linear():
    print("test_anthracene_linear ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), (1, 2), value=F_FUSED)
    M = compute_atom_count(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(M, 14, "anthracene atoms")
    bc, _ = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 16, "anthracene aromatic bonds")
    degrees = (bc > 0).sum(axis=1)
    assert_eq(int((degrees[:14] == 3).sum()), 4, "anthracene 4 deg-3")
    print("  PASS")


def test_5_6_fused_aromatic():
    print("test_5_6_fused_aromatic ...")
    R = _R_pad(RING_5_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    M = compute_atom_count(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(M, 9, "indole skeleton atoms")
    bc, _ = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 10, "indole skeleton aromatic")
    print("  PASS")


def test_cyclohexane():
    print("test_cyclohexane ...")
    R = _R_pad(RING_6_ALIPH)
    bc, _ = build_bond_classes(R, _zero_F(), _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_SINGLE).sum() // 2), 6, "cyclohexane 6 single")
    print("  PASS")


def test_biphenyl():
    print("test_biphenyl ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_LINKED)
    L = _L_with((0, 1, 0))
    M = compute_atom_count(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(M, 12, "biphenyl atoms")
    bc, _ = build_bond_classes(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 12, "biphenyl 12 aromatic")
    assert_eq(int((bc == BOND_SINGLE).sum() // 2), 1, "biphenyl 1 single")
    print("  PASS")


def test_diphenylmethane():
    print("test_diphenylmethane ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_LINKED)
    L = _L_with((0, 1, 1))
    M = compute_atom_count(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(M, 13, "diphenylmethane atoms")
    bc, _ = build_bond_classes(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 12, "diphenylmethane 12 aromatic")
    assert_eq(int((bc == BOND_SINGLE).sum() // 2), 2, "diphenylmethane 2 single")
    print("  PASS")


def test_fused_plus_linked():
    print("test_fused_plus_linked ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM, RING_6_AROM)
    F = np.zeros((R_MAX, R_MAX), dtype=np.int64)
    F[0, 1] = F[1, 0] = F_FUSED
    F[1, 2] = F[2, 1] = F_LINKED
    L = _L_with((1, 2, 0))
    M = compute_atom_count(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(M, 16, "1-phenylnaphthalene atoms")
    bc, _ = build_bond_classes(R, F, L, _zero_P_len(), _zero_P_pos())
    assert_eq(int((bc == BOND_AROMATIC).sum() // 2), 17, "phenylnaph aromatic")
    assert_eq(int((bc == BOND_SINGLE).sum() // 2), 1, "phenylnaph single")
    print("  PASS")


def test_aromaticity_mismatch_fusion_accepts_with_aromatic_precedence():
    """Naphthoquinone-class case: aromatic 6-ring fused to aliphatic 6-ring.
    The shared edge bond was previously rejected (Aromaticity mismatch).
    With Batch 1.2's aromatic precedence rule, the shared edge takes the
    AROMATIC class (since at least one ring is aromatic), and the aliphatic
    ring's other bonds remain SINGLE.

    This is the common naphthoquinone / indanone fusion pattern in RedDB.
    """
    print("test_aromaticity_mismatch_fusion_accepts_with_aromatic_precedence ...")
    R = _R_pad(RING_6_AROM, RING_6_ALIPH)
    F = _F_with((0, 1), value=F_FUSED)

    # Should NOT raise (was the previous behavior)
    bc, am = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())

    # Verify M_total
    M = compute_atom_count(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    assert_eq(M, 10, "naphthoquinone-skeleton 10 atoms")

    # Bond counts: aromatic 6-ring contributes 6 aromatic bonds.
    # The aliphatic 6-ring shares the fusion edge (which is now AROMATIC)
    # plus 5 other edges that are SINGLE. So total: 6 aromatic + 5 single.
    n_arom = int((bc == BOND_AROMATIC).sum() // 2)
    n_single = int((bc == BOND_SINGLE).sum() // 2)
    assert_eq(n_arom, 6, "6 aromatic bonds (full aromatic ring including fusion)")
    assert_eq(n_single, 5, "5 single bonds (aliphatic ring's non-fused edges)")
    print("  PASS")


def test_aromaticity_precedence_ring_order_independence():
    """Order should not matter: aliphatic-then-aromatic should give the
    same result as aromatic-then-aliphatic. Both should give aromatic
    precedence at the shared edge."""
    print("test_aromaticity_precedence_ring_order_independence ...")

    # Aromatic first
    R1 = _R_pad(RING_6_AROM, RING_6_ALIPH)
    F1 = _F_with((0, 1), value=F_FUSED)
    bc1, _ = build_bond_classes(R1, F1, _zero_L(), _zero_P_len(), _zero_P_pos())

    # Aliphatic first (so the SINGLE class is assigned BEFORE AROMATIC tries to override)
    R2 = _R_pad(RING_6_ALIPH, RING_6_AROM)
    F2 = _F_with((0, 1), value=F_FUSED)
    bc2, _ = build_bond_classes(R2, F2, _zero_L(), _zero_P_len(), _zero_P_pos())

    # Both should have 6 aromatic + 5 single bonds total
    n_arom_1 = int((bc1 == BOND_AROMATIC).sum() // 2)
    n_arom_2 = int((bc2 == BOND_AROMATIC).sum() // 2)
    n_single_1 = int((bc1 == BOND_SINGLE).sum() // 2)
    n_single_2 = int((bc2 == BOND_SINGLE).sum() // 2)
    assert_eq(n_arom_1, 6, "ring 0 aromatic case: 6 aromatic")
    assert_eq(n_arom_2, 6, "ring 0 aliphatic case: 6 aromatic (precedence works regardless of order)")
    assert_eq(n_single_1, 5, "ring 0 aromatic case: 5 single")
    assert_eq(n_single_2, 5, "ring 0 aliphatic case: 5 single")
    print("  PASS")


def test_pure_aromatic_fusion_unchanged():
    """Sanity: naphthalene (both rings aromatic) should still produce
    11 aromatic bonds. The aromatic-precedence rule should not change
    behavior when there's no mismatch."""
    print("test_pure_aromatic_fusion_unchanged ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    bc, _ = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    n_arom = int((bc == BOND_AROMATIC).sum() // 2)
    assert_eq(n_arom, 11, "naphthalene 11 aromatic bonds (unchanged from before)")
    print("  PASS")


def test_pure_aliphatic_fusion_unchanged():
    """Sanity: decalin (both rings aliphatic) should still produce
    11 single bonds. Aromatic-precedence rule has no effect when neither
    is aromatic."""
    print("test_pure_aliphatic_fusion_unchanged ...")
    R = _R_pad(RING_6_ALIPH, RING_6_ALIPH)
    F = _F_with((0, 1), value=F_FUSED)
    bc, _ = build_bond_classes(R, F, _zero_L(), _zero_P_len(), _zero_P_pos())
    n_single = int((bc == BOND_SINGLE).sum() // 2)
    assert_eq(n_single, 11, "decalin 11 single bonds (unchanged from before)")
    print("  PASS")


def test_disconnected_rings_rejected():
    print("test_disconnected_rings_rejected ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    try:
        compute_atom_count(R, _zero_F(), _zero_L(), _zero_P_len(), _zero_P_pos())
        raise AssertionError("Should have raised")
    except ValueError as e:
        if "no anchor" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_empty_layout():
    print("test_empty_layout ...")
    M = compute_atom_count(
        _R_pad(), _zero_F(), _zero_L(), _zero_P_len(), _zero_P_pos()
    )
    assert_eq(M, 0, "empty layout")
    print("  PASS")


# ════════════════════════════════════════════════════════════════════
# New: pendant with explicit P_pos
# ════════════════════════════════════════════════════════════════════

def test_pendant_at_explicit_position():
    """Benzene with pendant of length 2 at position 3.
    Verify atom 3 of ring 0 is the attachment (not some auto-chosen position)."""
    print("test_pendant_at_explicit_position ...")
    R = _R_pad(RING_6_AROM)
    F = _zero_F()
    L = _zero_L()
    P_len = _P_len_with((0, 0, 2))     # ring 0, slot 0, length 2
    P_pos = _P_pos_with((0, 0, 3))     # at position 3 of ring 0

    M = compute_atom_count(R, F, L, P_len, P_pos)
    assert_eq(M, 8, "benzene + ethyl atoms")

    layout = build_atom_layout(R, F, L, P_len, P_pos)
    pend = layout["pendant_atoms"][(0, 0)]
    attach_pos, atoms = pend
    assert_eq(attach_pos, 3, "pendant attaches at position 3")
    # Ring 0 is just [0,1,2,3,4,5]; atom at position 3 is atom 3
    # New pendant atoms get next indices: 6, 7

    bc, _ = build_bond_classes(R, F, L, P_len, P_pos)
    # Atom 3 should be bonded to atom 6 (single)
    assert_eq(int(bc[3, 6]), BOND_SINGLE, "atom 3 → 6 single bond")
    assert_eq(int(bc[6, 7]), BOND_SINGLE, "atom 6 → 7 single bond (chain)")
    # Atom 3 has degree 3 (2 aromatic neighbors in ring + 1 single to pendant)
    degrees = (bc > 0).sum(axis=1)
    assert_eq(int(degrees[3]), 3, "attach atom 3 has degree 3")
    print("  PASS")


def test_two_pendants_at_different_positions():
    """Benzene with pendants at positions 1 and 4 (opposite-ish)."""
    print("test_two_pendants_at_different_positions ...")
    R = _R_pad(RING_6_AROM)
    F = _zero_F()
    L = _zero_L()
    P_len = _P_len_with((0, 0, 1), (0, 1, 1))
    P_pos = _P_pos_with((0, 0, 1), (0, 1, 4))

    M = compute_atom_count(R, F, L, P_len, P_pos)
    assert_eq(M, 8, "benzene + 2 methyl atoms")

    layout = build_atom_layout(R, F, L, P_len, P_pos)
    pend0 = layout["pendant_atoms"][(0, 0)]
    pend1 = layout["pendant_atoms"][(0, 1)]
    assert_eq(pend0[0], 1, "pendant 0 at position 1")
    assert_eq(pend1[0], 4, "pendant 1 at position 4")

    bc, _ = build_bond_classes(R, F, L, P_len, P_pos)
    degrees = (bc > 0).sum(axis=1)
    # Atoms 1 and 4 (the attachment atoms) have degree 3
    assert_eq(int(degrees[1]), 3, "attach atom 1 has degree 3")
    assert_eq(int(degrees[4]), 3, "attach atom 4 has degree 3")
    print("  PASS")


def test_pendant_collision_with_fusion_rejected():
    """Naphthalene fusion uses positions (2, 3) of ring 0. Asking pendant
    on ring 0 to attach at position 2 should raise."""
    print("test_pendant_collision_with_fusion_rejected ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    L = _zero_L()
    P_len = _P_len_with((0, 0, 1))
    P_pos = _P_pos_with((0, 0, 2))   # collides with fusion edge

    try:
        compute_atom_count(R, F, L, P_len, P_pos)
        raise AssertionError("Should have raised on fusion collision")
    except ValueError as e:
        if "structural" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_pendant_collision_with_linker_rejected():
    """Biphenyl uses position 2 of ring 0 (the far edge first position) for
    the linker. Asking pendant on ring 0 to attach at position 2 should raise."""
    print("test_pendant_collision_with_linker_rejected ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_LINKED)
    L = _L_with((0, 1, 0))
    P_len = _P_len_with((0, 0, 1))
    P_pos = _P_pos_with((0, 0, 2))   # collides with linker attachment

    try:
        compute_atom_count(R, F, L, P_len, P_pos)
        raise AssertionError("Should have raised on linker collision")
    except ValueError as e:
        if "structural" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_two_pendants_same_position_rejected():
    """Two pendants on the same ring must not share a position."""
    print("test_two_pendants_same_position_rejected ...")
    R = _R_pad(RING_6_AROM)
    F = _zero_F()
    L = _zero_L()
    P_len = _P_len_with((0, 0, 1), (0, 1, 1))
    P_pos = _P_pos_with((0, 0, 3), (0, 1, 3))   # same position

    try:
        compute_atom_count(R, F, L, P_len, P_pos)
        raise AssertionError("Should have raised on duplicate pendant position")
    except ValueError as e:
        if "another pendant" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_pendant_pos_out_of_range_rejected():
    """P_pos must be < ring_size when P_len > 0."""
    print("test_pendant_pos_out_of_range_rejected ...")
    R = _R_pad(RING_6_AROM)   # size 6, valid positions 0-5
    F = _zero_F()
    L = _zero_L()
    P_len = _P_len_with((0, 0, 1))
    P_pos = _P_pos_with((0, 0, 7))   # 7 is out of range

    try:
        compute_atom_count(R, F, L, P_len, P_pos)
        raise AssertionError("Should have raised on out-of-range position")
    except ValueError as e:
        if "out of range" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_pendant_pos_nonzero_with_zero_length_rejected():
    """When P_len[i, p] == 0, P_pos[i, p] must also be 0 (PAD invariant)."""
    print("test_pendant_pos_nonzero_with_zero_length_rejected ...")
    R = _R_pad(RING_6_AROM)
    F = _zero_F()
    L = _zero_L()
    P_len = _P_len_with()  # all zeros
    P_pos = _P_pos_with((0, 0, 3))   # nonzero P_pos but P_len is 0

    try:
        compute_atom_count(R, F, L, P_len, P_pos)
        raise AssertionError("Should have raised on PAD invariant violation")
    except ValueError as e:
        if "P_len is 0" not in str(e):
            raise AssertionError(f"Wrong error: {e}")
    print("  PASS")


def test_naphthalene_with_pendant_at_position_5():
    """Naphthalene with pendant on ring 0 at position 5 (the position
    'opposite' to the fusion). Position 5 is valid because fusion uses (2,3).

    Total: 10 ring atoms + 3 pendant atoms = 13. Naphthalene topology
    preserved (11 aromatic bonds + 2 single from ring-pendant attachment
    + 2 internal pendant single bonds = 14 bonds total)."""
    print("test_naphthalene_with_pendant_at_position_5 ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    L = _zero_L()
    P_len = _P_len_with((0, 0, 3))
    P_pos = _P_pos_with((0, 0, 5))   # position 5 on ring 0 — far from fusion

    M = compute_atom_count(R, F, L, P_len, P_pos)
    assert_eq(M, 13, "naphthalene + propyl atoms")

    layout = build_atom_layout(R, F, L, P_len, P_pos)
    pend = layout["pendant_atoms"][(0, 0)]
    attach_pos, atoms = pend
    assert_eq(attach_pos, 5, "pendant at position 5")
    # Ring 0 is [0,1,2,3,4,5]; position 5 = atom 5
    # New pendant atoms: 10, 11, 12

    bc, _ = build_bond_classes(R, F, L, P_len, P_pos)
    assert_eq(int(bc[5, 10]), BOND_SINGLE, "atom 5 → 10 single bond (pendant attach)")
    print("  PASS")


def test_list_valid_pendant_positions_benzene():
    """For an isolated benzene (no fusion, no linker), all 6 positions valid."""
    print("test_list_valid_pendant_positions_benzene ...")
    R = _R_pad(RING_6_AROM)
    F = _zero_F()
    L = _zero_L()
    valid = list_valid_pendant_positions(R, F, L, ring_idx=0)
    assert_eq(sorted(valid), [0, 1, 2, 3, 4, 5], "all 6 positions valid for isolated benzene")
    print("  PASS")


def test_list_valid_pendant_positions_naphthalene_ring0():
    """For ring 0 of naphthalene, fusion uses positions (2, 3); others valid."""
    print("test_list_valid_pendant_positions_naphthalene_ring0 ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_FUSED)
    L = _zero_L()
    valid = list_valid_pendant_positions(R, F, L, ring_idx=0)
    # Fusion uses positions 2 and 3 (the far edge of ring 0)
    expected = [0, 1, 4, 5]
    assert_eq(sorted(valid), expected, "ring 0 of naphthalene valid pendant positions")
    print("  PASS")


def test_list_valid_pendant_positions_biphenyl_ring0():
    """For ring 0 of biphenyl, linker attaches at position 2 (the first atom
    of the far edge). Other 5 positions valid."""
    print("test_list_valid_pendant_positions_biphenyl_ring0 ...")
    R = _R_pad(RING_6_AROM, RING_6_AROM)
    F = _F_with((0, 1), value=F_LINKED)
    L = _L_with((0, 1, 0))
    valid = list_valid_pendant_positions(R, F, L, ring_idx=0)
    # Linker uses only position 2
    expected = [0, 1, 3, 4, 5]
    assert_eq(sorted(valid), expected, "ring 0 of biphenyl valid pendant positions")
    print("  PASS")


def main():
    print("=" * 70)
    print("HiMoFlow v5.4 Batch 1.1 — ring_layout_decoder tests (P_pos)")
    print("=" * 70)
    # Carry-forward
    test_benzene()
    test_naphthalene()
    test_anthracene_linear()
    test_5_6_fused_aromatic()
    test_cyclohexane()
    test_biphenyl()
    test_diphenylmethane()
    test_fused_plus_linked()
    test_aromaticity_mismatch_fusion_accepts_with_aromatic_precedence()
    test_aromaticity_precedence_ring_order_independence()
    test_pure_aromatic_fusion_unchanged()
    test_pure_aliphatic_fusion_unchanged()
    test_disconnected_rings_rejected()
    test_empty_layout()
    # New: explicit P_pos
    test_pendant_at_explicit_position()
    test_two_pendants_at_different_positions()
    test_pendant_collision_with_fusion_rejected()
    test_pendant_collision_with_linker_rejected()
    test_two_pendants_same_position_rejected()
    test_pendant_pos_out_of_range_rejected()
    test_pendant_pos_nonzero_with_zero_length_rejected()
    test_naphthalene_with_pendant_at_position_5()
    test_list_valid_pendant_positions_benzene()
    test_list_valid_pendant_positions_naphthalene_ring0()
    test_list_valid_pendant_positions_biphenyl_ring0()
    print()
    print("All 25 tests PASSED")


if __name__ == "__main__":
    main()
