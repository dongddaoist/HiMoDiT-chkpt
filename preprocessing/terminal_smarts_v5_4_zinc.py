"""
HiMoFlow v5.4-ZINC — Extended terminal SMARTS list (K=22).
# v5.4-ZINC Phase 2D applied. K=16 -> K=22 with 5 SMARTS repairs and 6 new terminals.

Phase 2D-A REPAIRS (charge-form bug fixes — same IDs):
  id=0  OH    add alkoxide (-O-) variant
  id=1  COOH  add carboxylate (-C(=O)O-) variant
  id=2  NH2   add ammonium (-NH3+) variant
  id=3  SO3H  add sulfonate (-S(=O)(=O)O-) variant
  id=13 NO2   change recursive 1-atom SMARTS to non-recursive 3-atom form
              (recursive form silently failed in detect_terminals_v3 because
               the crossing-bond logic needs the SMARTS match to enclose
               all 3 nitro atoms, not just the central N)

Phase 2D-C ADDITIONS (new IDs 16-21):
  id=16 Thiol            -SH                ~1-2% of ZINC
  id=17 AcylHalide       -C(=O)X            ~0.3% (default emit Cl)
  id=18 Cyanate          -OC#N              <0.1%
  id=19 Thiocyanate      -SC#N              <0.1%
  id=20 Isothiocyanate   -N=C=S             <0.1%
  id=21 Isonitrile       -[N+]#[C-]         <0.1%
"""
from __future__ import annotations

CURATED_TERMINALS = [
    # ─── v5.3 locked vocab (IDs 0-5) ─── Phase 2D-A REPAIRS applied to 0,1,2,3
    {"name": "OH", "detection_smarts": "[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 0,
     "notes": "Phase 2D-A: now matches both -OH and alkoxide -[O-]."},
    {"name": "COOH", "detection_smarts": "[CX3](=[OX1])[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 1,
     "notes": "Phase 2D-A: now matches both -C(=O)OH and carboxylate -C(=O)[O-]."},
    {"name": "NH2", "detection_smarts": "[NX3H2,NX4H3+]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 2,
     "notes": "Phase 2D-A: now matches both -NH2 and protonated -[NH3+]."},
    {"name": "SO3H", "detection_smarts": "[SX4](=[OX1])(=[OX1])[OX2H1,OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 3,
     "notes": "Phase 2D-A: now matches both -S(=O)(=O)OH and sulfonate -S(=O)(=O)[O-]."},
    {"name": "F", "detection_smarts": "[F]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 4},
    {"name": "CH3", "detection_smarts": "[CX4H3]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 5},

    # ─── v5.4 double-bond entries (IDs 6-8) ──────────────────────────────
    {"name": "=O", "detection_smarts": "[OX1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 6,
     "notes": "Carbonyl oxygen. Common in quinones, ketones."},
    {"name": "=NH", "detection_smarts": "[ND1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 7,
     "notes": "Imine nitrogen. [ND1] correctly rejects internal "
              "imines R-N=R\' (D=2)."},
    {"name": "=S", "detection_smarts": "[SX1]",
     "allowed_attachment_bonds": [3], "flags": [], "v5_3_id": 8,
     "notes": "Thione sulfur."},

    # ─── v5.4-ZINC additions (IDs 9-15) ──────────────────────────────────
    {"name": "Cl", "detection_smarts": "[Cl]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 9,
     "notes": "Chlorine. Single-atom terminal."},
    {"name": "Br", "detection_smarts": "[Br]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 10,
     "notes": "Bromine. Single-atom terminal."},
    {"name": "I", "detection_smarts": "[I]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 11,
     "notes": "Iodine. Single-atom terminal."},
    {"name": "CN", "detection_smarts": "[CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 12,
     "notes": "Nitrile group. 2-atom terminal."},
    # Phase 2D-A NO2 REPAIR — non-recursive 3-atom form so the crossing-bond
    # logic in _is_terminal_match correctly identifies the fragment boundary.
    {"name": "NO2", "detection_smarts": "[NX3+](=[OX1])[OX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 13,
     "notes": "Phase 2D-A: 3-atom SMARTS replaces broken 1-atom recursive form. "
              "Matches RDKit canonical zwitterion [N+](=O)[O-] form."},
    {"name": "OCH3", "detection_smarts": "[OX2][CX4H3]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 14,
     "notes": "Methoxy group. 2-atom terminal."},
    {"name": "CF3", "detection_smarts": "[CX4]([F])([F])[F]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 15,
     "notes": "Trifluoromethyl group. 4-atom terminal."},

    # ─── v5.4-ZINC Phase 2D-C additions (IDs 16-21) ──────────────────────
    {"name": "Thiol", "detection_smarts": "[SX2H][#6]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 16,
     "notes": "Phase 2D-C: thiol -SH attached to scaffold C."},
    {"name": "AcylHalide", "detection_smarts": "[CX3](=[OX1])[F,Cl,Br,I]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 17,
     "notes": "Phase 2D-C: acyl halide -C(=O)X. Default emit -C(=O)Cl."},
    {"name": "Cyanate", "detection_smarts": "[OX2][CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 18,
     "notes": "Phase 2D-C: cyanate -OC#N."},
    {"name": "Thiocyanate", "detection_smarts": "[SX2][CX2]#[NX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 19,
     "notes": "Phase 2D-C: thiocyanate -SC#N."},
    {"name": "Isothiocyanate", "detection_smarts": "[NX2]=[CX2]=[SX1]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 20,
     "notes": "Phase 2D-C: isothiocyanate -N=C=S."},
    {"name": "Isonitrile", "detection_smarts": "[NX2+]#[CX1-]",
     "allowed_attachment_bonds": [1], "flags": [], "v5_3_id": 21,
     "notes": "Phase 2D-C: isonitrile -[N+]#[C-]."},
]


def _validate_smarts():
    from rdkit import Chem
    failed = []
    for t in CURATED_TERMINALS:
        patt = Chem.MolFromSmarts(t["detection_smarts"])
        if patt is None:
            failed.append((t["name"], t["detection_smarts"]))
    if failed:
        msg = "Curated SMARTS failed to compile:\n"
        for name, smarts in failed:
            msg += f"  {name}: {smarts}\n"
        raise RuntimeError(msg)


_validate_smarts()
