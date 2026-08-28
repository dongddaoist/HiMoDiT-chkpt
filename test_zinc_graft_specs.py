"""
Round-trip test: take a benzene scaffold, graft each of the new K=16
terminal classes (10-16) onto the 6 ring atoms, and check that
assembly + RDKit sanitize succeed for every new fragment.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from meanflow.compose_full_molecule_zinc import assemble_to_smiles

# Build a benzene scaffold by hand: 6 aromatic C atoms, ring bonds aromatic
M_MAX = 24
atom_ids = np.zeros(M_MAX, dtype=np.int64)
atom_ids[:6] = 1  # vocab ID for 'c' (aromatic carbon)
atom_mask = np.zeros(M_MAX, dtype=bool); atom_mask[:6] = True
bond_classes = np.zeros((M_MAX, M_MAX), dtype=np.int64)
for i in range(6):
    j = (i + 1) % 6
    bond_classes[i, j] = 2; bond_classes[j, i] = 2  # aromatic

CLASS_NAMES = {
    1:'OH', 2:'COOH', 3:'NH2', 4:'SO3H', 5:'F', 6:'CH3',
    7:'=O', 8:'=NH', 9:'=S',
    10:'Cl', 11:'Br', 12:'I', 13:'CN',
    14:'NO2', 15:'OCH3', 16:'CF3',
}

print("Round-trip test: graft each terminal class on benzene atom 0.\n")
print(f"{'class':>5}  {'name':<6}  {'SMILES':<35}  {'sanitize':<8}")
print("─" * 70)
for cls, name in CLASS_NAMES.items():
    fragment_ids = np.zeros(M_MAX, dtype=np.int64)
    fragment_ids[0] = cls
    smi = assemble_to_smiles(atom_ids, bond_classes, atom_mask, fragment_ids)
    status = "PASS" if smi is not None else "FAIL"
    smi_str = smi or "(None)"
    print(f"{cls:>5}  {name:<6}  {smi_str:<35}  {status}")

# Also: graft Cl on all 6 atoms → hexachlorobenzene
print()
fragment_ids = np.zeros(M_MAX, dtype=np.int64)
fragment_ids[:6] = 10  # Cl
smi = assemble_to_smiles(atom_ids, bond_classes, atom_mask, fragment_ids)
print(f"6× Cl on benzene → {smi}")

# Multi-terminal: NO2 + OCH3 + CF3 on different atoms
fragment_ids = np.zeros(M_MAX, dtype=np.int64)
fragment_ids[0] = 14  # NO2
fragment_ids[2] = 15  # OCH3
fragment_ids[4] = 16  # CF3
smi = assemble_to_smiles(atom_ids, bond_classes, atom_mask, fragment_ids)
print(f"1,3,5-substituted (NO2, OCH3, CF3) → {smi}")
