"""
Reuse the diagnose_zinc250k_retention.py SMILES list, but inject extra
SMARTS terminals to verify that adding Cl/Br/CN/NO2/etc. as terminals
recovers the rejected molecules — no atom-vocab extension needed.
"""
import sys; sys.path.insert(0, '.')
import importlib

# Inject extra terminals into CURATED_TERMINALS BEFORE preprocessing
# imports (so the module-level _COMPILED_TERMINAL_PATTERNS is built
# with the extended list).
from preprocessing import terminal_smarts_v3_extended as tsv3

EXTRA_TERMINALS = [
    {"name": "Cl", "detection_smarts": "[Cl]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 9},
    {"name": "Br", "detection_smarts": "[Br]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 10},
    {"name": "I",  "detection_smarts": "[I]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 11},
    # Cyano: C#N. The C has D=2 (one bond to scaffold, one triple to N
    # which counts as 1 connection). [CX2]#[NX1] matches both atoms.
    {"name": "CN", "detection_smarts": "[CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 12},
    # Nitro: -[N+](=O)[O-] (RDKit canonical) or -N(=O)=O.
    # RDKit standard form is the zwitterion. Use $() to match either.
    {"name": "NO2", "detection_smarts": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 13},
    # Methoxy: -O-CH3. 2-atom terminal.
    {"name": "OCH3", "detection_smarts": "[OX2][CX4H3]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 14},
    # Trifluoromethyl: -C(F)(F)F. 4-atom terminal.
    {"name": "CF3", "detection_smarts": "[CX4]([F])([F])[F]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 15},
]

tsv3.CURATED_TERMINALS = tsv3.CURATED_TERMINALS + EXTRA_TERMINALS

# Now reload preprocessing modules so they pick up the extended list.
import preprocessing.ring_layout_dataset as rld
importlib.reload(rld)

# Re-import the diagnose loop with the patched module
from preprocessing.ring_layout_dataset import extract_layout
from collections import Counter, defaultdict
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Same SMILES list as before
SAMPLE_SMILES = [
    "CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "CN(C)CCC=C1c2ccccc2CCc2ccccc21", "OC(=O)c1ccccc1Nc1ccccc1Cl",
    "ClC1=C(Cl)C(=O)c2ccccc2C1=O", "Brc1ccc(/C=C/c2ccccc2)cc1",
    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC", "O=[N+]([O-])c1ccc(N)cc1",
    "N#Cc1ccc(C#N)cc1", "FC(F)(F)Oc1ccccc1", "COc1ccc(N)cc1OC",
    "CCOC(=O)c1ccc(N)cc1", "CN1CCN(c2ccc(C(=O)NC3CCN(Cc4ccccc4)CC3)cc2)CC1",
    "Cc1ccc(C(=O)Nc2ccc(S(N)(=O)=O)cc2)cc1", "CC(C)c1ccc(C(C)C(=O)NCC2CCCCN2)cc1",
    "O=C1CCCN1c1ccc2nc(N)nc(N)c2c1", "CC1(C)Nc2ccccc2NC1c1ccc(F)cc1",
    "C1CCNCC1", "C1COCCN1", "C1CCCCC1", "C1CC1", "C1CCC1", "C1CCCCCC1",
    "OC12CC3CC(C1)CC(C2)C3", "C1CC2(CC1)CCCCC2",
    "CN(C)CCN1c2ccccc2Sc2ccc(Cl)cc21",
    "Clc1ccc2c(c1)C(c1ccncc1)=NCC2", "c1ccc2c(c1)cc1ccc3ccccc3c1c2",
    "CCCCCCCCCC(=O)O", "CCCCCCCCNc1ccccc1", "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
    "BrCC(Br)Br", "c1ccc(Cc2ccccc2)cc1", "c1ccc(/C=C/c2ccccc2)cc1",
    "c1ccc(Sc2ccccc2)cc1", "Cn1ncc(Br)c1C", "CC(=O)c1ccc(NC(=O)C2CC2)cc1",
    "CN(C)C(=O)c1ccc(C(=O)NC2CCCCC2)cc1", "CCN(CC)C(=O)CN1CCN(c2ccccc2OC)CC1",
    "Cc1ccc(NC(=O)CSc2nnc(C)n2-c2ccccc2)cc1", "Cc1cccc(NC(=O)c2cc(C)on2)c1",
    "O=C(c1ccccc1)N1CCN(C(=O)c2ccccc2)CC1", "CC(C)Cn1ncc2c1NCN2c1ccc(Cl)cc1",
    "CC(C)CN1CCN(c2ccc(C#N)cc2)CC1", "Cc1cc(NC(=O)c2cccc(F)c2)on1",
    "Brc1ccc(CN2CCN(c3ccccc3)CC2)cc1", "OC(=O)CN1C(=O)c2ccccc2C1=O",
    "ClC(Cl)(Cl)C(O)C(Cl)(Cl)Cl", "CSc1nc(N)nc(N)n1", "CCOC(=O)c1ccccn1",
    "Nc1nccc(-c2ccccn2)n1", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O",
    "OC[C@H]1OC(O)C(O)C(O)C1O",
]


def diagnose(smiles_list):
    rejection = Counter()
    rejection_examples = defaultdict(list)
    n_kept = 0
    n_total = len(smiles_list)
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rejection["smiles_parse_failed"] += 1
            continue
        try:
            label, reason = extract_layout(smi)
        except Exception as e:
            reason = f"exception_{type(e).__name__}"
            label = None
        if label is None:
            bucket = reason or "unknown_None"
            for prefix in ("ring_", "too_many_rings_", "pendant_too_long_",
                           "atom_", "rings_"):
                if reason and reason.startswith(prefix):
                    if "spiro_invalid" in reason: bucket = "rings_spiro_invalid"
                    elif "peri_invalid" in reason: bucket = "rings_peri_invalid"
                    elif "too_many_rings" in reason: bucket = "too_many_rings"
                    elif "pendant_too_long" in reason: bucket = "pendant_too_long"
                    elif "size" in reason and "ring" in reason:
                        bucket = "ring_size_outside_5_6"
                    elif "not_in_vocab" in reason:
                        bucket = "atom_not_in_vocab"
                    break
            rejection[bucket] += 1
            if len(rejection_examples[bucket]) < 3:
                rejection_examples[bucket].append((smi, reason))
        else:
            n_kept += 1
    return n_kept, n_total, rejection, rejection_examples


print(f"\nWith EXTENDED terminals "
      f"(Cl, Br, I, CN, NO2, OCH3, CF3 added → K=16):")
print(f"  CURATED_TERMINALS now has {len(tsv3.CURATED_TERMINALS)} entries\n")
n_kept, n_total, rej, ex = diagnose(SAMPLE_SMILES)
print(f"Retention: {n_kept}/{n_total} ({100*n_kept/n_total:.0f}%)\n")

print(f"{'count':>5}  {'%-rej':>5}  reason")
print("─" * 60)
n_rej = n_total - n_kept
for reason, cnt in sorted(rej.items(), key=lambda kv: -kv[1]):
    pct = 100 * cnt / max(n_rej, 1)
    print(f"{cnt:>5}  {pct:>4.0f}%  {reason}")

print("\nRemaining rejection examples:")
for reason, cnt in sorted(rej.items(), key=lambda kv: -kv[1]):
    print(f"\n  [{cnt}] {reason}")
    for smi, full_reason in ex[reason]:
        print(f"    {smi:<55s}  ({full_reason})")
