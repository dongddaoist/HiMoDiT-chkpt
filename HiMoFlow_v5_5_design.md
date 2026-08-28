#HiMoDiT — design + current status

**Last updated**: 2026-05-11
**Status**: Encoder + decoder validated end-to-end. Training scripts and generation pipeline need integration work.
**Headline result**: 93.25% ZINC250K retention (vs 83.14% v5.4 baseline, +10.11 pp, 0 regressions).

---

## 1. What v5.5 is

v5.5 is the production-ready successor to v5.4-ZINC250K. It addresses three architectural limitations that caused v5.4 to reject ~17% of ZINC250K molecules:

1. **Linear-pendant constraint**: v5.4 rejected any pendant that branched (e.g. N,N-diethylamine → ✗). v5.5 represents pendants as TREES with parent indices, accepting arbitrary branched side-chains up to size 15.
2. **Spiro reject**: v5.4 rejected all ring pairs sharing one atom as `spiro_invalid`. v5.5 accepts spiro at sp³-quaternary centers (C, N+, Si) by adding `F_SPIRO=3` to the F-vocab.
3. **M_MAX hardening**: v5.4 had a default-arg staleness bug in `ring_layout_decoder.py` that caused ~28K (45% of failures) `M_MAX_overflow` rejections even though the runtime constant was bumped. v5.5 has the AST-level fix baked into source.

v5.5 changes the label schema (replacing P_len/P_pos with B_size/B_pos/B_parent/B_bond and adding spiro_atom_positions), but keeps the v5.4 atom vocab (K=16), terminal SMARTS (K=22), and ring-type vocab (K=11) — those Phase 2D additions are now permanent.

## 2. What's done in v5.5

### Source code (in this directory)

- ✅ `preprocessing/ring_layout_dataset.py`: appended `extract_layout_v5_5()` — the unified encoder. v5.4's `extract_layout()` is preserved alongside for diff and rollback.
- ✅ `preprocessing/terminal_smarts_v3_extended.py`: now re-exports K=22 from `terminal_smarts_v5_4_zinc.py`. No more in-memory monkey-patch needed.
- ✅ `meanflow/ring_layout_decoder.py`: added `F_SPIRO=3`, `B_LEN_MAX=15` constants, expanded F-value validation to accept F_SPIRO, appended `decode_v5_5_to_scaffold()` for the new label schema. M_MAX default-arg staleness bug fully patched.
- ✅ `meanflow/ring_atom_diffusion.py`: M_MAX now imported from `ring_layout_decoder` (single source of truth).
- ✅ `meanflow/compose_full_molecule_zinc.py`: K=16 atom vocab + K=22 terminals (unchanged from v5.4 Phase 2D).
- ✅ `run_training_v5_5_a1.py`, `run_training_v5_5_a2.py`, `run_training_v5_5_terminal.py`: renamed from v5.4, with header notes describing v5.5 changes needed before retraining.
- ✅ `run_training_v5_5_a3.py`: STUB (placeholder for the new branch-topology stage).
- ✅ `v5_5_diagnostics/`: archival of all prototype + diagnostic scripts that produced the validation numbers.

### Validation (production-scale)

| Encoder | 20K ZINC sample acceptance | Regressions |
|---------|---------------------------:|------------:|
| v5.4 baseline | 83.14% | — |
| v5.5 branched alone | 91.58% | 0 |
| v5.5 spiro alone | 89.83% | 0 |
| **v5.5 combined (this codebase)** | **93.25%** | **0** |

Combined recovery on 60,346-row failure CSV: **72.42%** (43,704 of 60,346 rejections recovered).

Per-bucket recovery of original failures:
- `M_MAX_overflow`: 100% (19,983 / 19,983) — hardening
- `decoder_roundtrip`: 99.7% (7,617 / 7,637) — hardening cascade
- `pendant_branched`: 99.5% (8,452 / 8,491) — branched encoder
- `pendant_position_invalid`: 99.3% (3,239 / 3,261) — branched encoder
- `pendant_too_long`: 100% (10 / 10) — branched encoder (B_LEN_MAX=15)
- `spiro_invalid`: 90.9% (4,377 / 4,814) — spiro encoder

Persistent rejections (deliberately not addressed in v5.5):
- `peri_fusion` / `peri_invalid`: 9,818 — high risk to lift, defer
- `angular_fusion`: 2,887 — could investigate, low priority
- `no_clean_endpoint`: 1,189 — v5.4 traversal edge case
- `no_rings_chain_only`: 1,109 — architectural, skip
- Ring sizes 8/9/10+ / too_many_rings: ~700 — not worth retraining

### Encode→decode roundtrip

Verified locally on 7 diverse test cases including spiro[5.5]undecanone, branched amines, naphthalene (fused), and a 24-atom drug-like. All produce valid bond_classes matrices with correct atom counts and reasonable bond counts (M-1 tree bonds + ring closures + branch tree edges).

## 3. What needs the next session

### A. Training-side integration 

Three things, each independent:

1. **A1 model F-classes**: The RingLayoutDiffusion model in `meanflow/ring_layout_diffusion.py` has an F-prediction head with `out_dim=3` (NONE/FUSED/LINKED). For v5.5 it needs `out_dim=4` to include F_SPIRO. Find and change that constant.
2. **Dataset .pkl extraction**: Run `extract_layout_v5_5()` over ZINC250K to produce a new labels .pkl. The notebook setup is identical to v5.4's batch-2.1 preprocessing — just swap the function name.
3. **A2 training**: No code changes required. Just point `--labels-pkl` at the v5.5 .pkl and retrain. The atom_ids+M_total fields are unchanged in shape.

### B. A3 (branch topology) implementation  

This is the genuinely new modeling work. The branch trees need a generative model. Two design options spelled out in `run_training_v5_5_a3.py`:

- **Two-headed transformer**: slot head for (B_size, B_pos) + autoregressive tree head for (B_parent, B_bond). Cleaner.
- **Flat MLP head**: predict all four tensors jointly. Simpler, worth trying first.

The A3 model takes input from A1+A2 (R, F, L, atom_ids, spiro_atom_positions) plus the property condition, and outputs (B_size, B_pos, B_parent, B_bond).

### C. Generation pipeline  

`meanflow/compose_full_molecule_zinc.py` currently assembles molecules from `(atom_ids, bond_classes, atom_mask, fragment_ids)`. The v5.5 generation path is:

1. Sample condition → A1 produces (R, F, L, spiro_atom_positions).
2. A2 produces atom_ids.
3. A3 produces (B_size, B_pos, B_parent, B_bond).
4. `decode_v5_5_to_scaffold()` consumes all of the above → bond_classes.
5. Terminal stage grafts substituents.
6. `assemble_to_smiles()` (existing) produces SMILES.

The new step is step 3. `decode_v5_5_to_scaffold()` is already written (tested for roundtrip). The plumbing in compose_full_molecule_zinc.py needs to be extended to accept v5.5 label tensors and call the new decoder.

### D. Notebooks  

Three notebooks to author, mirroring v5.4:

1. **`HiMoFlow_v5_5_preprocess_ZINC250K.ipynb`** — Drive setup, runs `extract_layout_v5_5()` over ZINC250K, exports the .pkl. Should show retention stats matching the 93.25% production number.
2. **`HiMoFlow_v5_5_train_and_eval.ipynb`** — runs all four trainers (a1, a2, a3, terminal), evaluates the trained model on held-out ZINC, runs `controllability_eval_zinc.py` for property-conditioning checks.
3. **`HiMoFlow_v5_5_generation.ipynb`** — sampling pipeline, full molecule generation, validity/uniqueness/property metrics.

## 4. File structure

```
mean-flow-v5.5-ZINC250K/
├── HiMoFlow_v5_5_design.md          ← THIS FILE (status + restart prompt)
├── meanflow/
│   ├── __init__.py
│   ├── ring_layout_decoder.py       ← F_SPIRO=3, decode_v5_5_to_scaffold added
│   ├── ring_layout_diffusion.py     ← TODO: bump F_classes 3→4
│   ├── ring_atom_diffusion.py       ← M_MAX import from decoder
│   ├── terminal_fragment_diffusion.py
│   ├── edge_biased_attention.py
│   ├── compose_full_molecule.py
│   └── compose_full_molecule_zinc.py ← TODO: hook decode_v5_5_to_scaffold
├── preprocessing/
│   ├── __init__.py
│   ├── ring_layout_dataset.py       ← extract_layout_v5_5() appended
│   ├── detect_terminals_v3.py
│   ├── terminal_smarts_v3_extended.py ← now re-exports K=22 vocab
│   └── terminal_smarts_v5_4_zinc.py  ← K=22 vocab (single source of truth)
├── run_training_v5_5_a1.py          ← runs unchanged; needs F_classes=4 in model
├── run_training_v5_5_a2.py          ← no changes needed
├── run_training_v5_5_terminal.py    ← no changes needed
├── run_training_v5_5_a3.py          ← STUB (next session)
├── compose_full_molecule_zinc.py    ← K=22 graft specs (Phase 2D)
├── controllability_eval_zinc.py     ← may need v5.5 label compatibility check
├── diagnose_zinc250k_retention.py
├── diagnose_zinc250k_retention_extended.py
├── tests/                            ← v5.4 test suite (still applicable to non-changed code)
├── v5_5_diagnostics/                 ← validation scripts (don't run during normal training)
│   ├── m_max_hardening.py           ← (already applied; idempotent)
│   ├── zinc_atom_inventory.py
│   ├── zinc_atom_inventory_scaffold.py
│   ├── analyze_combined.py
│   ├── analyze_failures.py
│   ├── prototype_branched_encoder.py
│   └── prototype_spiro_encoder.py
└── archive_v5_4/                     ← v5.4 notebooks + README + design doc, for reference
```

## 5. Production decision log

These decisions were validated by full-ZINC250K production runs:

- **ATOM_VOCAB stays at K=16.** The scaffold-aware inventory showed P (neutral) is the only missing-from-vocab atom in scaffold positions (~125 atoms, 0.002% of total). The cost/benefit doesn't warrant a vocab bump + retrain.
- **Cl/Br/I are terminal-only.** They appear in 56K atoms but 100% are in terminal positions (already handled by the K=22 terminal SMARTS as IDs 9/10/11). No scaffold vocab slot needed.
- **Peri-fusion stays rejected.** Lifting the tree constraint on the ring graph risks A1 quality regressions. The 9,818 affected molecules (~3.9% of ZINC) are not worth the risk.
- **No_rings_chain_only stays rejected.** ~0.4% of ZINC and requires architectural changes (the entire pipeline pivots around R[0]≠0). Defer.
- **R_MAX stays at 6.** Bumping to 8 affects 161 / 60,346 = 0.27% of failures.

## 6. Restart prompt for next Claude chat

When you start a new Claude conversation to continue this work, paste the following:

---


> The full design + status doc with file structure is at `HiMoDiT_design.md` in the v5.5 directory. Read it first before doing anything else.
>
> **What's blocking generation** (each ~1-3 days):
>
> 1. **A1 model bump**: `meanflow/ring_layout_diffusion.py` has `F_classes=3` somewhere — needs to be 4 to predict F_SPIRO.
> 2. **Run preprocessing**: `extract_layout_v5_5()` over ZINC250K → new .pkl with v5.5 labels.
> 3. **A3 (NEW)**: branch-topology diffusion stage. `run_training_v5_5_a3.py` is a stub with full design notes; implement the two-headed transformer or flat-MLP head as described.
> 4. **Compose update**: `meanflow/compose_full_molecule_zinc.py` needs to call `decode_v5_5_to_scaffold()` instead of v5.4's `decode_layout_to_scaffold()`.
> 5. **Generation notebooks**: preprocess, train, generate notebooks mirroring v5.4's three.
>
> **Working principles** that apply to all work on this project:
> - Chemistry-grounded design: hierarchical decomposition + sampler priorities must reflect how chemists actually design molecules. Algorithmic shortcuts that violate chemical plausibility are rejected even when convenient.
> - Diagnostic-first debugging: use targeted diagnostics (like `analyze_combined.py`) to isolate failure modes before committing to architectural changes.
> - Ablation discipline: each architectural change is validated independently before being combined.
> - Production-scale validation: every change is verified on full ZINC250K or the 60K failure CSV, not just on a few hand-picked SMILES.
> - Disk-commit caution: AST-based patchers (not text-replace) for any disk modification, idempotent.
>
> **Resources**:
> - Compute is plentiful — don't let perf constrain design.
> - ZINC CSV at `/content/drive/My Drive/machine-learning/Data/ZIN250K/250k_rndm_zinc_drugs_clean_3.csv`.
> - v5.4 codebase archived in `archive_v5_4/` for reference.
>
> **Next concrete step I want to take**: [USER fills this in based on what they want to tackle first — pre-preprocessing, A1 retrain, A3 implementation, or compose pipeline]

---

End of design doc.
