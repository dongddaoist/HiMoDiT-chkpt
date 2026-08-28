"""
HiMoFlow v5.4 — tests for preprocessing/ring_layout_dataset.

Verifies extraction correctness on canonical molecules, filter behavior
(reject phenanthrene/pyrene/spiro/branched), and round-trip validity
(decoded layout matches the original molecule's bond counts).

Run:  cd v5_4_batch2 && PYTHONPATH=. python tests/test_ring_layout_dataset.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from preprocessing.ring_layout_dataset import (
    extract_layout,
    _classify_ring_pair,
    _are_edges_opposite,
    _check_topology,
    _canonical_ring_order,
    _rdkit_atom_to_vocab_id,
    ATOM_VOCAB,
)
from meanflow.ring_layout_decoder import (
    decode_layout_to_scaffold,
    BOND_AROMATIC, BOND_SINGLE,
    RING_6_AROM, RING_6_ALIPH, RING_5_AROM, RING_5_ALIPH,
    F_FUSED, F_LINKED,
)
from rdkit import Chem


def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"{msg}: expected {b}, got {a}")


# ════════════════════════════════════════════════════════════════════
# Successful extraction tests (round-trip)
# ════════════════════════════════════════════════════════════════════

def _round_trip_check(smi, exp_M, exp_arom, exp_single):
    """Extract, decode, verify atom + bond counts match expectation."""
    label, reason = extract_layout(smi)
    if label is None:
        raise AssertionError(f"Extract failed for {smi}: {reason}")
    aip, bc, am = decode_layout_to_scaffold(
        label['R'], label['F'], label['L'],
        label['P_len'], label['P_pos'],
        label['atom_ids'],
    )
    n_arom = int((bc == BOND_AROMATIC).sum() // 2)
    n_single = int((bc == BOND_SINGLE).sum() // 2)
    M = label['M_total']
    if (M, n_arom, n_single) != (exp_M, exp_arom, exp_single):
        raise AssertionError(
            f"{smi}: round-trip mismatch. "
            f"got M={M},arom={n_arom},single={n_single} "
            f"expected M={exp_M},arom={exp_arom},single={exp_single}"
        )
    return label


def test_benzene():
    print("test_benzene ...")
    label = _round_trip_check('c1ccccc1', 6, 6, 0)
    assert_eq(int(label['R'][0]), RING_6_AROM, "benzene ring type")
    print("  PASS")


def test_naphthalene():
    print("test_naphthalene ...")
    label = _round_trip_check('c1ccc2ccccc2c1', 10, 11, 0)
    assert_eq(int(label['R'][0]), RING_6_AROM, "naph ring 0 type")
    assert_eq(int(label['R'][1]), RING_6_AROM, "naph ring 1 type")
    assert_eq(int(label['F'][0, 1]), F_FUSED, "naph fused")
    print("  PASS")


def test_biphenyl():
    print("test_biphenyl ...")
    label = _round_trip_check('c1ccc(-c2ccccc2)cc1', 12, 12, 1)
    assert_eq(int(label['F'][0, 1]), F_LINKED, "biphenyl linked")
    assert_eq(int(label['L'][0, 1]), 0, "biphenyl L=0")
    print("  PASS")


def test_diphenylmethane():
    print("test_diphenylmethane ...")
    label = _round_trip_check('c1ccc(Cc2ccccc2)cc1', 13, 12, 2)
    assert_eq(int(label['F'][0, 1]), F_LINKED, "diphenylmethane linked")
    assert_eq(int(label['L'][0, 1]), 1, "diphenylmethane L=1")
    print("  PASS")


def test_toluene():
    """Toluene `Cc1ccccc1` — CH3 stripped as terminal; scaffold = benzene only."""
    print("test_toluene ...")
    label = _round_trip_check('Cc1ccccc1', 6, 6, 0)
    # No pendant; CH3 is now a terminal
    assert_eq(int(label['P_len'][0, 0]), 0, "toluene CH3 stripped, no pendant")
    assert_eq(len(label['terminals']), 1, "1 CH3 terminal")
    print("  PASS")


def test_methylnaphthalene():
    """1-methylnaphthalene — CH3 stripped; scaffold = naphthalene only."""
    print("test_methylnaphthalene ...")
    label = _round_trip_check('Cc1cccc2ccccc21', 10, 11, 0)
    assert_eq(int(label['P_len'][0, 0]), 0, "CH3 stripped, no pendant")
    assert_eq(len(label['terminals']), 1, "1 CH3 terminal")
    print("  PASS")


def test_ethylbenzene():
    """Ethylbenzene `CCc1ccccc1` — terminal CH3 stripped; residual -CH2-
    remains as a length-1 pendant on the benzene ring."""
    print("test_ethylbenzene ...")
    label = _round_trip_check('CCc1ccccc1', 7, 6, 1)
    # Residual: 1 -CH2- atom as length-1 pendant
    assert_eq(int(label['P_len'][0, 0]), 1, "ethylbenzene 1 length-1 pendant (CH2)")
    assert_eq(len(label['terminals']), 1, "1 CH3 terminal stripped")
    print("  PASS")


def test_pyridine():
    print("test_pyridine ...")
    label = _round_trip_check('c1ccncc1', 6, 6, 0)
    # 1 nitrogen (vocab ID 5 = 'n') and 5 carbons (vocab ID 1 = 'c')
    n_count = int((label['atom_ids'] == 5).sum())
    c_count = int((label['atom_ids'] == 1).sum())
    assert_eq(n_count, 1, "pyridine has 1 aromatic N")
    assert_eq(c_count, 5, "pyridine has 5 aromatic C")
    print("  PASS")


def test_indole_skeleton():
    print("test_indole_skeleton ...")
    label = _round_trip_check('c1ccc2[nH]ccc2c1', 9, 10, 0)
    print("  PASS")


def test_terphenyl():
    print("test_terphenyl ...")
    label = _round_trip_check('c1ccc(-c2ccc(-c3ccccc3)cc2)cc1', 18, 18, 2)
    # 3 rings, all linked
    assert_eq(int(label['R'][0]), RING_6_AROM, "terphenyl ring 0")
    assert_eq(int(label['R'][1]), RING_6_AROM, "terphenyl ring 1")
    assert_eq(int(label['R'][2]), RING_6_AROM, "terphenyl ring 2")
    print("  PASS")


def test_cyclohexane():
    print("test_cyclohexane ...")
    label = _round_trip_check('C1CCCCC1', 6, 0, 6)
    assert_eq(int(label['R'][0]), RING_6_ALIPH, "cyclohexane is aliphatic")
    print("  PASS")


# ════════════════════════════════════════════════════════════════════
# Filter tests (must reject)
# ════════════════════════════════════════════════════════════════════

def test_phenanthrene_rejected():
    print("test_phenanthrene_rejected ...")
    label, reason = extract_layout('c1ccc2ccc3ccccc3c2c1')
    assert label is None, "phenanthrene should be rejected"
    assert "angular" in reason, f"reason should mention angular, got: {reason}"
    print(f"  PASS (rejected as: {reason})")


def test_pyrene_rejected():
    print("test_pyrene_rejected ...")
    label, reason = extract_layout('c1cc2ccc3cccc4ccc(c1)c2c34')
    assert label is None, "pyrene should be rejected"
    assert "cycle" in reason or "peri" in reason, (
        f"reason should mention cycle/peri, got: {reason}"
    )
    print(f"  PASS (rejected as: {reason})")


def test_spiro_rejected():
    print("test_spiro_rejected ...")
    label, reason = extract_layout('C1CCC2(CC1)CCCCC2')  # spiro[5.5]undecane
    assert label is None, "spiro should be rejected"
    assert "spiro" in reason, f"reason should mention spiro, got: {reason}"
    print(f"  PASS (rejected as: {reason})")


def test_cycloheptane_rejected():
    print("test_cycloheptane_rejected ...")
    label, reason = extract_layout('C1CCCCCC1')
    assert label is None, "cycloheptane should be rejected"
    assert "size" in reason, f"reason should mention size, got: {reason}"
    print(f"  PASS (rejected as: {reason})")


# ════════════════════════════════════════════════════════════════════
# Batch 2.1: terminal-stripping tests
# ════════════════════════════════════════════════════════════════════

def test_phenol_strips_OH():
    """Phenol → benzene scaffold + 1 OH terminal."""
    print("test_phenol_strips_OH ...")
    label, reason = extract_layout('Oc1ccccc1')
    assert label is not None, f"phenol should pass: {reason}"
    assert_eq(label['M_total'], 6, "phenol scaffold M (just benzene)")
    assert_eq(len(label['terminals']), 1, "phenol has 1 terminal")
    assert_eq(label['terminals'][0]['name'], 'OH', "phenol terminal is OH")
    print("  PASS")


def test_benzoic_acid_strips_COOH():
    """Benzoic acid → benzene scaffold + 1 COOH terminal."""
    print("test_benzoic_acid_strips_COOH ...")
    label, reason = extract_layout('OC(=O)c1ccccc1')
    assert label is not None, f"benzoic acid should pass: {reason}"
    assert_eq(label['M_total'], 6, "benzoic acid scaffold M")
    assert_eq(len(label['terminals']), 1, "benzoic acid has 1 terminal")
    assert_eq(label['terminals'][0]['name'], 'COOH', "terminal is COOH")
    print("  PASS")


def test_4_aminobenzoic_acid_strips_two_terminals():
    """4-aminobenzoic acid → benzene scaffold + COOH + NH2."""
    print("test_4_aminobenzoic_acid_strips_two_terminals ...")
    label, reason = extract_layout('Nc1ccc(C(=O)O)cc1')
    assert label is not None, f"4-aminobenzoic acid should pass: {reason}"
    assert_eq(label['M_total'], 6, "scaffold is just benzene")
    assert_eq(len(label['terminals']), 2, "2 terminals")
    names = sorted(t['name'] for t in label['terminals'])
    assert_eq(names, ['COOH', 'NH2'], "terminals are COOH+NH2")
    print("  PASS")


def test_dihydroxybenzoic_acid_three_terminals():
    """2,4-dihydroxybenzoic acid → benzene + 2 OH + 1 COOH (3 terminals)."""
    print("test_dihydroxybenzoic_acid_three_terminals ...")
    label, reason = extract_layout('OC(=O)c1ccc(O)cc1O')
    assert label is not None, f"should pass: {reason}"
    assert_eq(label['M_total'], 6, "scaffold is just benzene")
    assert_eq(len(label['terminals']), 3, "3 terminals")
    names = sorted(t['name'] for t in label['terminals'])
    assert_eq(names, ['COOH', 'OH', 'OH'], "terminals are 2 OH + 1 COOH")
    print("  PASS")


def test_cyclopentenedione_double_O_terminals():
    """Cyclopentenedione `O=C1C=CC(=O)C1` → 5-aliphatic ring + 2 =O terminals."""
    print("test_cyclopentenedione_double_O_terminals ...")
    label, reason = extract_layout('O=C1C=CC(=O)C1')
    assert label is not None, f"cyclopentenedione should pass: {reason}"
    assert_eq(label['M_total'], 5, "cyclopentenedione scaffold is 5-ring only")
    assert_eq(int(label['R'][0]), RING_5_ALIPH,
              "cyclopentenedione is aliphatic 5-ring")
    assert_eq(len(label['terminals']), 2, "2 =O terminals")
    names = [t['name'] for t in label['terminals']]
    assert names == ['=O', '=O'], f"both terminals are =O, got {names}"
    bond_classes_terminals = [t['attach_bond_class'] for t in label['terminals']]
    assert all(b == 3 for b in bond_classes_terminals), (
        "=O attaches by DOUBLE bond"
    )
    print("  PASS")


def test_benzoquinone_double_O_terminals():
    """1,4-benzoquinone → 6-aliphatic ring + 2 =O terminals."""
    print("test_benzoquinone_double_O_terminals ...")
    label, reason = extract_layout('O=C1C=CC(=O)C=C1')
    assert label is not None, f"benzoquinone should pass: {reason}"
    assert_eq(label['M_total'], 6, "benzoquinone scaffold is 6-ring only")
    assert_eq(len(label['terminals']), 2, "2 =O terminals")
    names = [t['name'] for t in label['terminals']]
    assert names == ['=O', '=O'], f"both =O, got {names}"
    print("  PASS")


def test_iminoquinone_O_and_NH():
    """Iminoquinone `O=C1C=CC(=N)C=C1` → ring + =O + =NH."""
    print("test_iminoquinone_O_and_NH ...")
    label, reason = extract_layout('O=C1C=CC(=N)C=C1')
    assert label is not None, f"iminoquinone should pass: {reason}"
    assert_eq(label['M_total'], 6, "iminoquinone scaffold is 6-ring")
    assert_eq(len(label['terminals']), 2, "2 terminals")
    names = sorted(t['name'] for t in label['terminals'])
    assert_eq(names, ['=NH', '=O'], "terminals are =O + =NH")
    print("  PASS")


def test_thiobenzoquinone_O_and_S():
    """Thiobenzoquinone → ring + =O + =S."""
    print("test_thiobenzoquinone_O_and_S ...")
    label, reason = extract_layout('O=C1C=CC(=S)C=C1')
    assert label is not None, f"thiobenzoquinone should pass: {reason}"
    assert_eq(label['M_total'], 6, "thiobenzoquinone scaffold")
    assert_eq(len(label['terminals']), 2, "2 terminals")
    names = sorted(t['name'] for t in label['terminals'])
    assert_eq(names, ['=O', '=S'], "terminals are =O + =S")
    print("  PASS")


def test_naphthol_strips_OH_keeps_naphthalene():
    """2-naphthol `Oc1ccc2ccccc2c1` → naphthalene + 1 OH."""
    print("test_naphthol_strips_OH_keeps_naphthalene ...")
    label, reason = extract_layout('Oc1ccc2ccccc2c1')
    assert label is not None, f"2-naphthol should pass: {reason}"
    assert_eq(label['M_total'], 10, "naphthalene scaffold")
    assert_eq(len(label['terminals']), 1, "1 OH terminal")
    print("  PASS")


def test_no_double_count_OH_in_COOH():
    """The OH inside COOH must NOT be detected as a separate terminal
    (parent-child subset removal handles this)."""
    print("test_no_double_count_OH_in_COOH ...")
    label, reason = extract_layout('OC(=O)c1ccccc1')
    assert label is not None
    # Should have exactly 1 terminal (COOH), NOT 2 (COOH + OH)
    assert_eq(len(label['terminals']), 1, "only COOH, OH is child of COOH")
    assert_eq(label['terminals'][0]['name'], 'COOH', "the terminal is COOH")
    print("  PASS")


def test_no_double_count_O_in_COOH():
    """The =O inside COOH must NOT be detected as a separate =O terminal."""
    print("test_no_double_count_O_in_COOH ...")
    label, reason = extract_layout('OC(=O)c1ccccc1')
    assert label is not None
    # Should have exactly 1 terminal (COOH), NOT a separate =O
    names = [t['name'] for t in label['terminals']]
    assert names == ['COOH'], f"COOH absorbs =O, got {names}"
    print("  PASS")


def test_methylnaphthalene_strips_methyl():
    """Methylnaphthalene → naphthalene + 1 CH3 terminal."""
    print("test_methylnaphthalene_strips_methyl ...")
    label, reason = extract_layout('Cc1cccc2ccccc21')
    assert label is not None, f"methylnaphthalene should pass: {reason}"
    assert_eq(label['M_total'], 10, "naphthalene scaffold")
    assert_eq(len(label['terminals']), 1, "1 CH3 terminal")
    assert_eq(label['terminals'][0]['name'], 'CH3', "terminal is CH3")
    print("  PASS")


def test_isopropylbenzene_handled():
    """Isopropylbenzene `CC(C)c1ccccc1` — both -CH3 are terminals (stripped).
    Remaining residual on the ring: 1 -CH- atom (length-1 pendant)."""
    print("test_isopropylbenzene_handled ...")
    label, reason = extract_layout('CC(C)c1ccccc1')
    if label is None:
        raise AssertionError(
            f"isopropylbenzene should now extract (terminals stripped): {reason}"
        )
    assert_eq(label['M_total'], 7, "isopropylbenzene scaffold M (benzene + CH residual)")
    assert_eq(int(label['P_len'][0, 0]), 1, "1 length-1 pendant")
    n_ch3 = sum(1 for t in label['terminals'] if t['name'] == 'CH3')
    assert_eq(n_ch3, 2, "2 CH3 terminals")
    print("  PASS")


def test_no_rings_rejected():
    print("test_no_rings_rejected ...")
    label, reason = extract_layout('CCCCC')  # pentane
    assert label is None, "pentane (no rings) should be rejected"
    assert "no_rings" in reason
    print(f"  PASS (rejected as: {reason})")


def test_invalid_smiles_rejected():
    print("test_invalid_smiles_rejected ...")
    label, reason = extract_layout('not_a_smiles')
    assert label is None, "invalid SMILES should be rejected"
    assert "parse" in reason
    print(f"  PASS (rejected as: {reason})")


# ════════════════════════════════════════════════════════════════════
# Helper function tests
# ════════════════════════════════════════════════════════════════════

def test_classify_ring_pair_naphthalene():
    print("test_classify_ring_pair_naphthalene ...")
    m = Chem.MolFromSmiles('c1ccc2ccccc2c1')
    rings = list(m.GetRingInfo().AtomRings())
    rel, info = _classify_ring_pair(m, 0, 1, rings)
    assert_eq(rel, "fused", "naphthalene rings classified as fused")
    assert_eq(len(info["shared_atoms"]), 2, "naphthalene fusion shares 2 atoms")
    print("  PASS")


def test_classify_ring_pair_biphenyl():
    print("test_classify_ring_pair_biphenyl ...")
    m = Chem.MolFromSmiles('c1ccc(-c2ccccc2)cc1')
    rings = list(m.GetRingInfo().AtomRings())
    rel, info = _classify_ring_pair(m, 0, 1, rings)
    assert_eq(rel, "linked", "biphenyl rings classified as linked")
    assert_eq(info["linker_length"], 0, "biphenyl L=0")
    print("  PASS")


def test_classify_ring_pair_diphenylmethane():
    print("test_classify_ring_pair_diphenylmethane ...")
    m = Chem.MolFromSmiles('c1ccc(Cc2ccccc2)cc1')
    rings = list(m.GetRingInfo().AtomRings())
    rel, info = _classify_ring_pair(m, 0, 1, rings)
    assert_eq(rel, "linked", "diphenylmethane rings linked")
    assert_eq(info["linker_length"], 1, "diphenylmethane L=1")
    assert_eq(len(info["linker_atoms"]), 1, "diphenylmethane has 1 linker atom")
    print("  PASS")


def test_are_edges_opposite_6ring():
    print("test_are_edges_opposite_6ring ...")
    ring = [10, 11, 12, 13, 14, 15]
    assert _are_edges_opposite(ring, {10, 11}, {13, 14}), "6-ring opposite edges"
    assert not _are_edges_opposite(ring, {10, 11}, {11, 12}), "6-ring adjacent edges"
    assert not _are_edges_opposite(ring, {10, 11}, {12, 13}), "6-ring almost-opposite"
    print("  PASS")


def test_atom_vocab_mapping():
    print("test_atom_vocab_mapping ...")
    m = Chem.MolFromSmiles('c1ccncc1')  # pyridine: 1 aromatic N + 5 aromatic C
    atoms = m.GetAtoms()
    n_count = sum(1 for a in atoms if _rdkit_atom_to_vocab_id(a) == 5)  # 'n'
    c_count = sum(1 for a in atoms if _rdkit_atom_to_vocab_id(a) == 1)  # 'c'
    assert_eq(n_count, 1, "pyridine 1 aromatic N")
    assert_eq(c_count, 5, "pyridine 5 aromatic C")
    print("  PASS")


# ════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("HiMoFlow v5.4 Batch 2.1 — ring_layout_dataset tests")
    print("=" * 70)
    # Round-trip extraction tests (no terminals)
    test_benzene()
    test_naphthalene()
    test_biphenyl()
    test_diphenylmethane()
    test_toluene()
    test_methylnaphthalene()
    test_ethylbenzene()
    test_pyridine()
    test_indole_skeleton()
    test_terphenyl()
    test_cyclohexane()
    # Filter tests
    test_phenanthrene_rejected()
    test_pyrene_rejected()
    test_spiro_rejected()
    test_cycloheptane_rejected()
    test_isopropylbenzene_handled()
    test_no_rings_rejected()
    test_invalid_smiles_rejected()
    # Helper tests
    test_classify_ring_pair_naphthalene()
    test_classify_ring_pair_biphenyl()
    test_classify_ring_pair_diphenylmethane()
    test_are_edges_opposite_6ring()
    test_atom_vocab_mapping()
    # Batch 2.1: terminal-stripping tests
    test_phenol_strips_OH()
    test_benzoic_acid_strips_COOH()
    test_4_aminobenzoic_acid_strips_two_terminals()
    test_dihydroxybenzoic_acid_three_terminals()
    test_cyclopentenedione_double_O_terminals()
    test_benzoquinone_double_O_terminals()
    test_iminoquinone_O_and_NH()
    test_thiobenzoquinone_O_and_S()
    test_naphthol_strips_OH_keeps_naphthalene()
    test_no_double_count_OH_in_COOH()
    test_no_double_count_O_in_COOH()
    test_methylnaphthalene_strips_methyl()
    print()
    print("All 35 tests PASSED")


if __name__ == "__main__":
    main()
