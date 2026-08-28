"""
HiMoFlow v5.4-ZINC — Deterministic scaffold + terminal composition (K=16).

Drop-in replacement for meanflow/compose_full_molecule.py extended for
the ZINC250K terminal vocabulary. Adds graft specs for IDs 9-15:
    Cl (id 9), Br (id 10), I (id 11), CN (id 12),
    NO2 (id 13), OCH3 (id 14), CF3 (id 15)
mapping to model classes 10-16 (model_class = SMARTS_id + 1).

Existing IDs 0-8 are preserved verbatim — Stage 2 weights trained on
the v5.4 K=9 vocab transfer cleanly into K=16 with zero-init for
heads 10-16.

USAGE
=====
  from meanflow.compose_full_molecule_zinc import (
      assemble_to_smiles, assemble_batch_to_smiles
  )
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType


# ─── Atom vocab (unchanged from v5.4 K=9) ────────────────────────────
# Halogens are NOT added here — they're handled as terminals only.
# v5.4-ZINC Phase 2D applied. ATOM_VOCAB 10 -> 16.
ATOM_VOCAB = ["<PAD>", "c", "O", "C", "N", "n", "S", "F", "s", "o", "O-", "N+", "n+", "N-", "n-", "P+"]

# Phase 2D-B: standard valence per atom symbol (for nH inference at
# charged-N/P emission). Used by vocab_id_to_smiles_atom().
_STD_VALENCE = {"N+": 4, "P+": 4, "N-": 2, "n+": 4, "n-": 2}

def vocab_id_to_smiles_atom(vid, degree=0):
    """Map vocab id -> SMILES atom token. Charged N/P need explicit nH from degree."""
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>": return ""
    if sym in ("c", "C", "O", "N", "n", "S", "F", "s", "o"): return sym
    if sym == "O-": return "[O-]"
    if sym in ("N+", "P+", "N-", "n+", "n-"):
        nh = max(0, _STD_VALENCE[sym] - max(0, degree))
        elem = sym[0]
        chg = "+" if "+" in sym else "-"
        if nh == 0: return "[" + elem + chg + "]"
        if nh == 1: return "[" + elem + "H" + chg + "]"
        return "[" + elem + "H" + str(nh) + chg + "]"
    raise ValueError("Unknown vocab symbol: " + repr(sym))


def _vocab_id_to_atom(vid: int) -> Tuple[str, bool]:
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>":
        raise ValueError("Cannot place PAD atom")
    if sym.islower():
        return sym.upper(), True
    return sym, False


_BOND_CLASS_TO_TYPE = {
    1: BondType.SINGLE,
    2: BondType.AROMATIC,
    3: BondType.DOUBLE,
    4: BondType.TRIPLE,
}


# ─── Terminal fragment graft specifications (K=16) ───────────────────
# Each entry:
#   atoms:  list of (element, is_aromatic, num_explicit_h)
#   bonds:  list of (a_idx, b_idx, BondType) — bonds INTERNAL to fragment
#   attach: BondType for the bond from scaffold host to fragment atom 0
#
# Atom 0 is the anchor — it bonds to the scaffold host. Internal bonds
# always reference previously-added atoms within the fragment.
#
# Model class index (1..16) → fragment spec.
_TERMINAL_SPECS = {
    # ───── v5.4 K=9 originals (unchanged) ─────
    1: dict(atoms=[("O", False, 1)], bonds=[], attach=BondType.SINGLE),
    2: dict(atoms=[("C", False, 0), ("O", False, 0), ("O", False, 1)],
            bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.SINGLE)],
            attach=BondType.SINGLE),
    3: dict(atoms=[("N", False, 2)], bonds=[], attach=BondType.SINGLE),
    4: dict(atoms=[("S", False, 0), ("O", False, 0), ("O", False, 0),
                   ("O", False, 1)],
            bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.DOUBLE),
                   (0, 3, BondType.SINGLE)],
            attach=BondType.SINGLE),
    5: dict(atoms=[("F", False, 0)], bonds=[], attach=BondType.SINGLE),
    6: dict(atoms=[("C", False, 3)], bonds=[], attach=BondType.SINGLE),
    7: dict(atoms=[("O", False, 0)], bonds=[], attach=BondType.DOUBLE),
    8: dict(atoms=[("N", False, 1)], bonds=[], attach=BondType.DOUBLE),
    9: dict(atoms=[("S", False, 0)], bonds=[], attach=BondType.DOUBLE),

    # ───── v5.4-ZINC additions (model classes 10-16) ─────
    # 10: Cl
    10: dict(atoms=[("Cl", False, 0)], bonds=[], attach=BondType.SINGLE),
    # 11: Br
    11: dict(atoms=[("Br", False, 0)], bonds=[], attach=BondType.SINGLE),
    # 12: I
    12: dict(atoms=[("I", False, 0)], bonds=[], attach=BondType.SINGLE),
    # 13: CN (-C#N). atom 0 = C (anchor), atom 1 = N. Internal C#N triple.
    13: dict(atoms=[("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    # 14: NO2 (-N(=O)=O, neutral form for assembly clarity).
    # atom 0 = N (anchor), atoms 1,2 = O. Internal N=O double bonds.
    # We use the neutral form here at assembly time; RDKit's sanitize
    # will normalize to the zwitterion canonical if needed.
    14: dict(atoms=[("N", False, 0), ("O", False, 0), ("O", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.DOUBLE)],
             attach=BondType.SINGLE),
    # 15: OCH3 (-O-CH3). atom 0 = O (anchor), atom 1 = C with 3 H.
    15: dict(atoms=[("O", False, 0), ("C", False, 3)],
             bonds=[(0, 1, BondType.SINGLE)],
             attach=BondType.SINGLE),
    # 16: CF3 (-C(F)(F)F). atom 0 = C (anchor), atoms 1,2,3 = F.
    16: dict(atoms=[("C", False, 0), ("F", False, 0),
                    ("F", False, 0), ("F", False, 0)],
             bonds=[(0, 1, BondType.SINGLE),
                    (0, 2, BondType.SINGLE),
                    (0, 3, BondType.SINGLE)],
             attach=BondType.SINGLE),

    # ───── v5.5 additions (model classes 17-22; SMARTS ids 16-21) ─────
    # 17: Thiol (-SH). Anchor S with 1 explicit H.
    17: dict(atoms=[("S", False, 1)], bonds=[], attach=BondType.SINGLE),
    # 18: AcylHalide (-C(=O)Cl). Default emits chloride per Phase 2D-C spec.
    # atom 0 = C (anchor), atom 1 = O (carbonyl), atom 2 = Cl.
    18: dict(atoms=[("C", False, 0), ("O", False, 0), ("Cl", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.SINGLE)],
             attach=BondType.SINGLE),
    # 19: Cyanate (-OC#N). atom 0 = O (anchor), atom 1 = C, atom 2 = N.
    19: dict(atoms=[("O", False, 0), ("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.SINGLE), (1, 2, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    # 20: Thiocyanate (-SC#N). atom 0 = S (anchor), atom 1 = C, atom 2 = N.
    20: dict(atoms=[("S", False, 0), ("C", False, 0), ("N", False, 0)],
             bonds=[(0, 1, BondType.SINGLE), (1, 2, BondType.TRIPLE)],
             attach=BondType.SINGLE),
    # 21: Isothiocyanate (-N=C=S). atom 0 = N (anchor), atom 1 = C, atom 2 = S.
    21: dict(atoms=[("N", False, 0), ("C", False, 0), ("S", False, 0)],
             bonds=[(0, 1, BondType.DOUBLE), (1, 2, BondType.DOUBLE)],
             attach=BondType.SINGLE),
    # 22: Isonitrile (-[N+]#[C-]). atom 0 = N+ (anchor), atom 1 = C-.
    # Use the formal-charge form; RDKit will normalize during sanitize.
    22: dict(atoms=[("N", False, 0), ("C", False, 0)],
             bonds=[(0, 1, BondType.TRIPLE)],
             attach=BondType.SINGLE),
}


def assemble_molecule(
    atom_ids: np.ndarray,
    bond_classes: np.ndarray,
    atom_mask: np.ndarray,
    fragment_ids: np.ndarray,
    sanitize: bool = True,
) -> Optional[Chem.Mol]:
    """Assemble (scaffold + terminals) into an RDKit Mol.

    Returns the sanitized Mol on success, None on any failure.
    """
    try:
        mol = Chem.RWMol()
        slot_to_rdkit = {}

        # 1. Scaffold atoms
        for i in range(len(atom_ids)):
            if not atom_mask[i]: continue
            vid = int(atom_ids[i])
            if vid == 0: return None
            element, is_arom = _vocab_id_to_atom(vid)
            atom = Chem.Atom(element)
            atom.SetIsAromatic(is_arom)
            slot_to_rdkit[i] = mol.AddAtom(atom)

        # 2. Scaffold bonds (upper triangle)
        for i in range(len(atom_ids)):
            if i not in slot_to_rdkit: continue
            for j in range(i + 1, len(atom_ids)):
                if j not in slot_to_rdkit: continue
                bc = int(bond_classes[i, j])
                if bc == 0: continue
                if bc not in _BOND_CLASS_TO_TYPE: return None
                mol.AddBond(slot_to_rdkit[i], slot_to_rdkit[j],
                            _BOND_CLASS_TO_TYPE[bc])

        # 3. Graft terminals
        for i in range(len(fragment_ids)):
            if not atom_mask[i]: continue
            fid = int(fragment_ids[i])
            if fid == 0: continue
            spec = _TERMINAL_SPECS.get(fid)
            if spec is None: return None
            frag_idx_to_rdkit = {}
            for k, (element, is_arom, num_h) in enumerate(spec["atoms"]):
                a = Chem.Atom(element)
                a.SetIsAromatic(is_arom)
                if num_h > 0: a.SetNumExplicitHs(num_h)
                frag_idx_to_rdkit[k] = mol.AddAtom(a)
            mol.AddBond(slot_to_rdkit[i], frag_idx_to_rdkit[0], spec["attach"])
            for (a, b, bt) in spec["bonds"]:
                mol.AddBond(frag_idx_to_rdkit[a], frag_idx_to_rdkit[b], bt)

        # 4. Sanitize
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
        return mol
    except Exception:
        return None


def assemble_to_smiles(
    atom_ids: np.ndarray,
    bond_classes: np.ndarray,
    atom_mask: np.ndarray,
    fragment_ids: np.ndarray,
) -> Optional[str]:
    mol = assemble_molecule(atom_ids, bond_classes, atom_mask, fragment_ids)
    if mol is None: return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def assemble_batch_to_smiles(
    atom_ids_b, bond_classes_b, atom_mask_b, fragment_ids_b,
):
    import torch
    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    a = _to_np(atom_ids_b); b = _to_np(bond_classes_b)
    m = _to_np(atom_mask_b); f = _to_np(fragment_ids_b)
    return [assemble_to_smiles(a[i], b[i], m[i], f[i]) for i in range(a.shape[0])]
