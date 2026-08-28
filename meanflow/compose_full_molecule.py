"""
HiMoFlow v5.4 — Deterministic scaffold + terminal composition.

Takes the output of A2 (scaffold atom_ids + bond_classes) and Stage 2
(per-atom fragment_ids) and assembles a complete RDKit molecule. This
is the deterministic graft step that closes the generative loop:

    (sol, gap) → A1 → decoder → A2 → Stage 2 → assemble_molecule → SMILES

INPUTS
------
  atom_ids     : (M_MAX,) long, in v5.3 atom vocab
                 0=<PAD>, 1=c, 2=O, 3=C, 4=N, 5=n, 6=S, 7=F, 8=s, 9=o
  bond_classes : (M_MAX, M_MAX) long, symmetric
                 0=none, 1=single, 2=aromatic
  atom_mask    : (M_MAX,) bool — real vs padding scaffold atoms
  fragment_ids : (M_MAX,) long, per-atom fragment class
                 0 = no decoration, 1..9 = OH..=S (model classes)

OUTPUT
------
  rdkit Mol if assembly + sanitize succeeds, else None.
  Use `Chem.MolToSmiles(mol)` to get the SMILES.

Failure modes (caller handles by counting None):
  - Invalid valence (e.g., =O on already-saturated atom)
  - Aromatic ring system not perceivable
  - Disconnected atoms / dangling bonds
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BondType


# ─── Atom vocab (must match preprocessing/ring_layout_dataset.py) ─────
ATOM_VOCAB = ["<PAD>", "c", "O", "C", "N", "n", "S", "F", "s", "o"]


# Map vocab id → (RDKit element, is_aromatic)
def _vocab_id_to_atom(vid: int) -> Tuple[str, bool]:
    sym = ATOM_VOCAB[vid]
    if sym == "<PAD>":
        raise ValueError("Cannot place PAD atom")
    if sym.islower():
        return sym.upper(), True   # aromatic
    return sym, False


# Map bond class → RDKit BondType
_BOND_CLASS_TO_TYPE = {
    1: BondType.SINGLE,
    2: BondType.AROMATIC,
    3: BondType.DOUBLE,    # used only for terminal attachment, not scaffold
    4: BondType.TRIPLE,    # unused in v5.4
}


# ─── Terminal fragment graft specifications ──────────────────────────
# Each entry: list of (element, is_aromatic, num_explicit_h) atoms +
# list of (a_idx, b_idx, BondType) bonds *internal* to the fragment +
# attach_bond_type to scaffold host.
#
# atom 0 of each fragment is the "anchor" — the atom that bonds to
# the scaffold host. Subsequent atoms are added with bonds *only* to
# previously-added atoms within the fragment.
#
# Model class index (1..9) → fragment spec.
_TERMINAL_SPECS = {
    # OH: [O]-H (the H is implicit/explicit per RDKit)
    1: dict(
        atoms=[("O", False, 1)],   # 1 explicit H
        bonds=[],
        attach=BondType.SINGLE,
    ),
    # COOH: -C(=O)OH
    2: dict(
        atoms=[("C", False, 0), ("O", False, 0), ("O", False, 1)],
        bonds=[(0, 1, BondType.DOUBLE), (0, 2, BondType.SINGLE)],
        attach=BondType.SINGLE,
    ),
    # NH2: -N(H)(H)
    3: dict(
        atoms=[("N", False, 2)],
        bonds=[],
        attach=BondType.SINGLE,
    ),
    # SO3H: -S(=O)(=O)OH
    4: dict(
        atoms=[("S", False, 0), ("O", False, 0), ("O", False, 0),
               ("O", False, 1)],
        bonds=[(0, 1, BondType.DOUBLE),
               (0, 2, BondType.DOUBLE),
               (0, 3, BondType.SINGLE)],
        attach=BondType.SINGLE,
    ),
    # F: -F
    5: dict(
        atoms=[("F", False, 0)],
        bonds=[],
        attach=BondType.SINGLE,
    ),
    # CH3: -C(H)(H)(H)
    6: dict(
        atoms=[("C", False, 3)],
        bonds=[],
        attach=BondType.SINGLE,
    ),
    # =O: =O (double bond from scaffold)
    7: dict(
        atoms=[("O", False, 0)],
        bonds=[],
        attach=BondType.DOUBLE,
    ),
    # =NH: =N(H)
    8: dict(
        atoms=[("N", False, 1)],
        bonds=[],
        attach=BondType.DOUBLE,
    ),
    # =S: =S
    9: dict(
        atoms=[("S", False, 0)],
        bonds=[],
        attach=BondType.DOUBLE,
    ),
}


def assemble_molecule(
    atom_ids: np.ndarray,        # (M_MAX,) long
    bond_classes: np.ndarray,    # (M_MAX, M_MAX) long
    atom_mask: np.ndarray,       # (M_MAX,) bool
    fragment_ids: np.ndarray,    # (M_MAX,) long
    sanitize: bool = True,
) -> Optional[Chem.Mol]:
    """Assemble (scaffold + terminals) into an RDKit Mol.

    Returns the sanitized Mol on success, None on any failure.
    Set sanitize=False to skip sanitization (returns the raw RWMol;
    useful for diagnostics).
    """
    try:
        mol = Chem.RWMol()
        # Map canonical scaffold slot → RDKit atom idx
        slot_to_rdkit = {}

        # 1. Add scaffold atoms.
        for i in range(len(atom_ids)):
            if not atom_mask[i]: continue
            vid = int(atom_ids[i])
            if vid == 0:
                # PAD inside the masked region is invalid — treat as failure.
                return None
            element, is_arom = _vocab_id_to_atom(vid)
            atom = Chem.Atom(element)
            atom.SetIsAromatic(is_arom)
            slot_to_rdkit[i] = mol.AddAtom(atom)

        # 2. Add scaffold bonds (upper triangle to avoid double-add).
        for i in range(len(atom_ids)):
            if i not in slot_to_rdkit: continue
            for j in range(i + 1, len(atom_ids)):
                if j not in slot_to_rdkit: continue
                bc = int(bond_classes[i, j])
                if bc == 0: continue
                if bc not in _BOND_CLASS_TO_TYPE:
                    return None
                mol.AddBond(slot_to_rdkit[i], slot_to_rdkit[j],
                            _BOND_CLASS_TO_TYPE[bc])

        # 3. Graft terminals.
        for i in range(len(fragment_ids)):
            if not atom_mask[i]: continue
            fid = int(fragment_ids[i])
            if fid == 0: continue
            spec = _TERMINAL_SPECS.get(fid)
            if spec is None: return None
            # Add atoms of the fragment
            frag_idx_to_rdkit = {}
            for k, (element, is_arom, num_h) in enumerate(spec["atoms"]):
                a = Chem.Atom(element)
                a.SetIsAromatic(is_arom)
                if num_h > 0: a.SetNumExplicitHs(num_h)
                frag_idx_to_rdkit[k] = mol.AddAtom(a)
            # Bond fragment anchor (idx 0) to scaffold host
            mol.AddBond(slot_to_rdkit[i], frag_idx_to_rdkit[0],
                        spec["attach"])
            # Internal fragment bonds
            for (a, b, bt) in spec["bonds"]:
                mol.AddBond(frag_idx_to_rdkit[a], frag_idx_to_rdkit[b], bt)

        # 4. Sanitize.
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
    """Convenience: assemble + Chem.MolToSmiles, returning canonical
    SMILES or None on failure."""
    mol = assemble_molecule(atom_ids, bond_classes, atom_mask, fragment_ids)
    if mol is None: return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def assemble_batch_to_smiles(
    atom_ids_b,           # (B, M_MAX) tensor or array
    bond_classes_b,       # (B, M_MAX, M_MAX) tensor or array
    atom_mask_b,          # (B, M_MAX) tensor or array
    fragment_ids_b,       # (B, M_MAX) tensor or array
):
    """Vectorized convenience: assemble each row, return list[Optional[str]]."""
    import torch
    def _to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    a = _to_np(atom_ids_b); b = _to_np(bond_classes_b)
    m = _to_np(atom_mask_b); f = _to_np(fragment_ids_b)
    smis = []
    for i in range(a.shape[0]):
        smis.append(assemble_to_smiles(a[i], b[i], m[i], f[i]))
    return smis
