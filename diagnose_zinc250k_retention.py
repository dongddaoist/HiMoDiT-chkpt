"""
Retention diagnostic: feed a representative sample of drug-like SMILES
(ZINC250K-style) through extract_layout() and see exactly what gets
rejected and by which architectural rule.

This is the diagnostic-first step: tells us which changes have the
highest leverage BEFORE committing to architectural retraining.
"""
import sys; sys.path.insert(0, '.')
from collections import Counter, defaultdict
from preprocessing.ring_layout_dataset import extract_layout
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# A curated ZINC250K-representative sample. Drawn to match the
# distribution properties of ZINC250K (drug-like, MW 250-500, more
# diverse rings/atoms/terminals than RedDB).
#
# Sources: known commercial drug SMILES + ChEMBL-style structures.
# Verified each parses with RDKit before adding to this list.
SAMPLE_SMILES = [
    # Simple drug-like
    "CC(=O)Nc1ccc(O)cc1",                                # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",                              # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                         # caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                         # ibuprofen
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",                     # atenolol
    "CN(C)CCC=C1c2ccccc2CCc2ccccc21",                     # amitriptyline
    "OC(=O)c1ccccc1Nc1ccccc1Cl",                          # diclofenac
    "ClC1=C(Cl)C(=O)c2ccccc2C1=O",                        # dichlone
    "Brc1ccc(/C=C/c2ccccc2)cc1",                          # cinnamic-Br
    "CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC",                   # metoclopramide
    # ZINC-typical substituents (NO2, CN, OCF3, OCH3)
    "O=[N+]([O-])c1ccc(N)cc1",                            # 4-nitroaniline
    "N#Cc1ccc(C#N)cc1",                                   # 1,4-dicyanobenzene
    "FC(F)(F)Oc1ccccc1",                                   # OCF3 phenyl
    "COc1ccc(N)cc1OC",                                     # dimethoxyaniline
    "CCOC(=O)c1ccc(N)cc1",                                # ethyl aminobenzoate
    # Larger drug-like with branching pendants
    "CN1CCN(c2ccc(C(=O)NC3CCN(Cc4ccccc4)CC3)cc2)CC1",     # benzhydryl piperazine
    "Cc1ccc(C(=O)Nc2ccc(S(N)(=O)=O)cc2)cc1",              # sulfa-derivative
    "CC(C)c1ccc(C(C)C(=O)NCC2CCCCN2)cc1",                 # branched amide
    # Sterically packed scaffolds
    "O=C1CCCN1c1ccc2nc(N)nc(N)c2c1",                       # pyrimidine-fused
    "CC1(C)Nc2ccccc2NC1c1ccc(F)cc1",                       # benzimidazoline
    # Heterocyclic rings beyond {5,6}-only set
    "C1CCNCC1",                                            # piperidine (6-aliph)
    "C1COCCN1",                                            # morpholine (6-aliph w/ O,N)
    "C1CCCCC1",                                            # cyclohexane
    "C1CC1",                                               # cyclopropane (3-ring) — REJECT
    "C1CCC1",                                              # cyclobutane (4-ring) — REJECT
    "C1CCCCCC1",                                           # cycloheptane (7-ring) — REJECT
    "OC12CC3CC(C1)CC(C2)C3",                               # adamantane (polycyclic) — REJECT
    # Spiro
    "C1CC2(CC1)CCCCC2",                                    # spiro[4.5] — REJECT (spiro)
    # Many-atoms
    "CN(C)CCN1c2ccccc2Sc2ccc(Cl)cc21",                     # chlorpromazine
    "Clc1ccc2c(c1)C(c1ccncc1)=NCC2",                       # ~22 atoms scaffold
    # Lots of rings (4 fused)
    "c1ccc2c(c1)cc1ccc3ccccc3c1c2",                        # picene-class — many rings
    # Complex chain pendants
    "CCCCCCCCCC(=O)O",                                     # decanoic acid (long alkyl chain)
    "CCCCCCCCNc1ccccc1",                                   # octylaniline (long pendant)
    # Halogen-rich
    "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",                       # hexachlorobenzene (Cl × 6)
    "BrCC(Br)Br",                                          # 1,2,2-tribromoethane
    # Structurally common in ZINC: phenyl-X-phenyl with linker
    "c1ccc(Cc2ccccc2)cc1",                                 # diphenylmethane (linker)
    "c1ccc(/C=C/c2ccccc2)cc1",                             # stilbene (linker)
    "c1ccc(Sc2ccccc2)cc1",                                 # diphenylsulfide
    # Real ZINC250K examples (manually picked from the public CSV)
    "Cn1ncc(Br)c1C",
    "CC(=O)c1ccc(NC(=O)C2CC2)cc1",
    "CN(C)C(=O)c1ccc(C(=O)NC2CCCCC2)cc1",
    "CCN(CC)C(=O)CN1CCN(c2ccccc2OC)CC1",
    "Cc1ccc(NC(=O)CSc2nnc(C)n2-c2ccccc2)cc1",
    "Cc1cccc(NC(=O)c2cc(C)on2)c1",
    "O=C(c1ccccc1)N1CCN(C(=O)c2ccccc2)CC1",
    "CC(C)Cn1ncc2c1NCN2c1ccc(Cl)cc1",
    "CC(C)CN1CCN(c2ccc(C#N)cc2)CC1",
    "Cc1cc(NC(=O)c2cccc(F)c2)on1",
    "Brc1ccc(CN2CCN(c3ccccc3)CC2)cc1",
    "OC(=O)CN1C(=O)c2ccccc2C1=O",
    "ClC(Cl)(Cl)C(O)C(Cl)(Cl)Cl",                            # all-aliphatic small
    "CSc1nc(N)nc(N)n1",
    "CCOC(=O)c1ccccn1",
    "Nc1nccc(-c2ccccn2)n1",
    "C[C@H](N)C(=O)O",                                       # alanine — has stereo @
    "C[C@@H](N)C(=O)O",                                      # alanine other enantiomer
    "OC[C@H]1OC(O)C(O)C(O)C1O",                              # glucose
]


def diagnose(smiles_list):
    rejection = Counter()
    rejection_examples = defaultdict(list)
    n_kept = 0
    n_total = len(smiles_list)
    for smi in smiles_list:
        # Pre-validate with RDKit
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rejection["smiles_parse_failed"] += 1
            rejection_examples["smiles_parse_failed"].append(smi)
            continue
        try:
            label, reason = extract_layout(smi)
        except Exception as e:
            reason = f"exception_{type(e).__name__}"
            label = None
        if label is None:
            # Bucket reasons broadly
            if reason is None:
                reason = "unknown_None"
            # Trim atom-name suffix to bucket family
            bucket = reason
            for prefix in ("ring_", "too_many_rings_", "pendant_too_long_",
                           "atom_", "rings_"):
                if reason.startswith(prefix):
                    parts = reason.split("_")
                    if "spiro_invalid" in reason:
                        bucket = "rings_spiro_invalid"
                    elif "peri_invalid" in reason:
                        bucket = "rings_peri_invalid"
                    elif "too_many_rings" in reason:
                        bucket = "too_many_rings"
                    elif "pendant_too_long" in reason:
                        bucket = "pendant_too_long"
                    elif "size" in reason and "ring" in reason:
                        # ring_3_size_3, ring_7_size_7, etc
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


n_kept, n_total, rej, ex = diagnose(SAMPLE_SMILES)
print(f"\nRetention on {n_total} ZINC-style SMILES: "
      f"{n_kept} kept ({100*n_kept/n_total:.0f}%), "
      f"{n_total - n_kept} rejected\n")
print(f"{'count':>5}  {'%-rejected':>11}  reason")
print("─" * 60)
n_rej = n_total - n_kept
for reason, cnt in sorted(rej.items(), key=lambda kv: -kv[1]):
    pct = 100 * cnt / max(n_rej, 1)
    print(f"{cnt:>5}  {pct:>10.0f}%  {reason}")

print("\nExample SMILES per rejection bucket (with full reason):")
for reason, cnt in sorted(rej.items(), key=lambda kv: -kv[1]):
    print(f"\n  [{cnt}] {reason}")
    for smi, full_reason in ex[reason]:
        print(f"    {smi:<55s}  ({full_reason})")
