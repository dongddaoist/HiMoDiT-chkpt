# HiMoDiT — ZINC250K-validated generative scaffold model

## Quick orientation

 

## File layout

See `HiMoDiT_design.md` §4 for the full file tree.

Quick highlights:
- `preprocessing/ring_layout_dataset.py` — has both `extract_layout()` (v5.4) and `extract_layout_()` (new). New code should use the v5.5 version.
- `meanflow/ring_layout_decoder.py` — has both `decode_layout_to_scaffold()` (v5.4) and `decode__to_scaffold()` (new). New code should use the v5.5 version.
- `_diagnostics/` — validation scripts that produced the 93.25% number. These were used to validate v5.5; they're not part of the runtime path.
- `archive_v5_4/` — v5.4 notebooks and README, kept for reference.

## How to use this codebase

```python
# Encode a SMILES → v5.5 label
from preprocessing.ring_layout_dataset import extract_layout_
label, reason = extract_layout_("CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br")
# label = {'R': ..., 'F': ..., 'L': ..., 'B_size': ..., ... 'M_total': 11}

# Decode label → bond_classes matrix
from meanflow.ring_layout_decoder import decode__to_scaffold
bond_classes, atom_mask = decode__to_scaffold(
    label['R'], label['F'], label['L'],
    label['B_size'], label['B_pos'], label['B_parent'], label['B_bond'],
    label['spiro_atom_positions'], label['atom_ids'],
)
```
