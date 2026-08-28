# ═══════════════════════════════════════════════════════════════════════
# 8b. Failure dumper — diagnose the 169 (assembly-failed) molecules
# ═══════════════════════════════════════════════════════════════════════
#
# What this cell does:
#   Re-runs the generation pipeline for the EXACT same 1024 samples whose
#   V·U·N was just computed (same seeds, same conditions), but this time
#   we capture all intermediate per-molecule tensors. For each None-SMILES
#   sample we record:
#     - the assembly-pre-sanitize SMILES (or "<unprintable>")
#     - the exact RDKit error class & message
#     - failure category (decode/grafting/sanitize)
#     - sanitize sub-bucket (kekulize / valence / element / etc.)
#     - scaffold metadata: n_active_atoms, n_aromatic_atoms, n_rings,
#       atom-id histogram, terminal-id histogram (which spec was grafted)
#     - the conditioning vector used
#
# Output:
#   - In-notebook grouped table by error bucket (top 5 examples each)
#   - JSON file at $BASE/failure_dump_v5_5_1.json for offline analysis
#
# What we're hoping to learn from the 169 failures:
#   Path A — Dominated by one sub-bucket (e.g., 80% are "Can't kekulize"
#            with 5-ring all-c scaffolds): single targeted patch.
#   Path B — Long tail with no single dominant pattern: validity ceiling
#            from this model is ~85% without retraining.
#   Path C — Cluster of failures involving a specific terminal class
#            (e.g., class 22 isonitrile): one-line spec fix.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import json
import os
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import RWMol, Atom

# Reuse pipeline internals
from meanflow.compose_full_molecule_zinc import (
    _vocab_id_to_atom, _vocab_id_to_charge, _BOND_CLASS_TO_TYPE,
    _TERMINAL_SPECS, ATOM_VOCAB, assemble_molecule,
)
from meanflow.ring_layout_decoder import (
    decode_v5_5_to_scaffold, aromatic_constraint_mask_v5_5,
    M_MAX, R_MAX, P_MAX, B_LEN_MAX,
)

# Sanity check: we need all_smiles and all_conds from the V·U·N cell.
assert len(all_smiles) == N_SAMPLES, (
    f"all_smiles has {len(all_smiles)} entries, expected {N_SAMPLES}. "
    f"Run the generation cell + V·U·N cell first."
)
assert all_conds.shape == (N_SAMPLES, 2), (
    f"all_conds shape {tuple(all_conds.shape)}, expected ({N_SAMPLES}, 2)"
)

# ─── Identify the failure indices ─────────────────────────────────────
failure_indices = [i for i, smi in enumerate(all_smiles) if smi is None]
n_failures = len(failure_indices)
print(f"Total failures to diagnose: {n_failures}/{N_SAMPLES} "
      f"({100*n_failures/N_SAMPLES:.1f}%)")

if n_failures == 0:
    print("No failures — nothing to dump.")
else:
    # ─── Re-run generation pipeline to recover intermediate tensors ────
    # We need atom_ids / bond_classes / fragment_ids per failed sample.
    # The generation cell only kept the SMILES; we have to regenerate.
    # Seeds match the V·U·N run exactly: SEED + batch_idx * 7 per batch.
    #
    # Note: this re-runs the FULL 1024, not just the failures — we can't
    # selectively run individual samples through the diffusion pipeline
    # because batches share random state. Re-runs cleanly because the
    # pipeline is deterministic given the seed.
    print(f"\nRe-running pipeline to recover intermediate tensors...")

    all_atom_ids       = []  # list of (M_MAX,) numpy arrays
    all_bond_classes   = []  # list of (M_MAX, M_MAX) numpy arrays
    all_atom_mask      = []  # list of (M_MAX,) bool numpy arrays
    all_fragment_ids   = []  # list of (M_MAX,) numpy arrays
    decode_failure_idx = set()  # global indices where decode_v5_5 raised

    @torch.no_grad()
    def regenerate_and_capture(cond_batch, seed):
        """Re-run pipeline for one batch, return (atom_ids, bond_classes,
        atom_mask, fragment_ids, decode_failed_indices_in_batch)."""
        B = cond_batch.shape[0]
        # A1
        a1_out = A1.sample(condition=cond_batch, n_steps=A1_STEPS,
                           temperature=TEMPERATURE, cfg_scale=CFG_SCALE, seed=seed)
        R, F_, L = a1_out['R'], a1_out['F'], a1_out['L']
        spc = a1_out['spiro_pos_class']
        # A3 spiro convention conversion
        a3_spiro = torch.where(spc == 0, torch.full_like(spc, 7), spc - 1)
        a3_out = A3.sample(R=R, F_mat=F_, L_mat=L, spiro_pos=a3_spiro,
                           condition=cond_batch, cfg_scale=CFG_SCALE,
                           temperature=TEMPERATURE, post_process=True,
                           seed=seed + 1)
        # Decoder spiro convention
        spiro_dec = torch.where(spc == 0, torch.full_like(spc, -1), spc - 1)

        bc_list, am_list, arm_list = [], [], []
        local_decode_fail = []
        aid_pl = np.zeros(M_MAX, dtype=np.int64)
        for i in range(B):
            try:
                bc, am = decode_v5_5_to_scaffold(
                    R[i].cpu().numpy(), F_[i].cpu().numpy(), L[i].cpu().numpy(),
                    a3_out['B_size'][i].cpu().numpy(),
                    a3_out['B_pos'][i].cpu().numpy(),
                    a3_out['B_parent'][i].cpu().numpy(),
                    a3_out['B_bond'][i].cpu().numpy(),
                    spiro_dec[i].cpu().numpy(), aid_pl, M_MAX_out=M_MAX,
                )
                arm = aromatic_constraint_mask_v5_5(bc, am)
                bc_list.append(bc); am_list.append(am); arm_list.append(arm)
            except Exception:
                local_decode_fail.append(i)
                bc_list.append(np.zeros((M_MAX, M_MAX), dtype=np.int64))
                am_list.append(np.zeros(M_MAX, dtype=bool))
                arm_list.append(np.zeros(M_MAX, dtype=bool))

        bond_classes = torch.from_numpy(np.stack(bc_list)).long().to(device)
        atom_mask    = torch.from_numpy(np.stack(am_list)).bool().to(device)
        arom_mask    = torch.from_numpy(np.stack(arm_list)).bool().to(device)

        atom_ids = A2.sample(bond_classes=bond_classes, atom_mask=atom_mask,
                             arom_mask=arom_mask, condition=cond_batch,
                             n_steps=A2_STEPS, temperature=TEMPERATURE,
                             cfg_scale=CFG_SCALE, seed=seed + 2)
        for i in local_decode_fail: atom_ids[i] = 0

        fragment_ids = TERM.sample(
            scaffold_atom_ids=atom_ids, scaffold_bond_classes=bond_classes,
            scaffold_atom_mask=atom_mask, condition=cond_batch,
            n_steps=TERM_STEPS, temperature=TEMPERATURE, seed=seed + 3,
        )
        return (atom_ids.cpu().numpy(), bond_classes.cpu().numpy(),
                atom_mask.cpu().numpy(), fragment_ids.cpu().numpy(),
                local_decode_fail)

    # Match the exact batching/seeding of the V·U·N generation cell:
    #   torch.manual_seed(SEED + 100); per batch: cond = randn(bs, 2);
    #   batch_smiles = generate_batch(cond, seed=SEED + batch_idx * 7)
    torch.manual_seed(SEED + 100)
    n_batches = (N_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    cursor = 0
    for batch_idx in range(n_batches):
        bs = min(BATCH_SIZE, N_SAMPLES - cursor)
        if bs <= 0: break
        cond_batch = torch.randn(bs, 2, device=device)
        a, b, m, f, ldf = regenerate_and_capture(cond_batch, seed=SEED + batch_idx * 7)
        for i in range(bs):
            all_atom_ids.append(a[i])
            all_bond_classes.append(b[i])
            all_atom_mask.append(m[i])
            all_fragment_ids.append(f[i])
            if i in ldf:
                decode_failure_idx.add(cursor + i)
        cursor += bs
    print(f"  Recovered tensors for {len(all_atom_ids)} samples; "
          f"{len(decode_failure_idx)} decode failures.")

    # ─── Sanity check: do the recovered SMILES match all_smiles? ───────
    # If the pipeline is deterministic given seed, this should match
    # exactly. A mismatch would indicate the failure dump is contaminated.
    mismatch = 0
    for i in range(N_SAMPLES):
        recovered = assemble_molecule(
            all_atom_ids[i], all_bond_classes[i],
            all_atom_mask[i], all_fragment_ids[i],
        )
        recovered_smi = Chem.MolToSmiles(recovered) if recovered else None
        # Compare against canonicalized all_smiles
        orig = all_smiles[i]
        if orig is None and recovered_smi is None:
            continue
        if orig is None or recovered_smi is None:
            mismatch += 1
            continue
        orig_canon = Chem.MolToSmiles(Chem.MolFromSmiles(orig)) if orig else None
        if orig_canon != recovered_smi:
            mismatch += 1
    if mismatch == 0:
        print(f"  ✓ All {N_SAMPLES} samples reproduced identically.")
    else:
        print(f"  ⚠ {mismatch} samples differ between V·U·N run and "
              f"failure-dump regeneration. Diagnoses below may be slightly "
              f"off if your run lands on a different RNG state. The error-"
              f"bucket distribution should still be representative.")

    # ─── Diagnose each failure ──────────────────────────────────────────
    print(f"\nDiagnosing {n_failures} failures...")
    failure_records = []

    for i in failure_indices:
        aid = all_atom_ids[i]
        bc  = all_bond_classes[i]
        am  = all_atom_mask[i]
        fid = all_fragment_ids[i]
        cond_i = all_conds[i].tolist()

        # Metadata
        n_active = int(am.sum())
        n_aromatic = int(((bc == 2).any(axis=1)).sum())  # rough: atoms with any aromatic bond
        n_bonds_total = int((bc > 0).sum() // 2)  # symmetric
        atom_hist = Counter(int(a) for j, a in enumerate(aid) if am[j])
        atom_hist_sym = {ATOM_VOCAB[k] if k < len(ATOM_VOCAB) else f"vid{k}": v
                         for k, v in atom_hist.items()}
        frag_hist = Counter(int(f) for j, f in enumerate(fid) if am[j] and f > 0)

        # Category 1: decode failure
        if i in decode_failure_idx:
            failure_records.append({
                'idx': i, 'category': 'decode_failed',
                'error_class': None, 'error_msg': 'decode_v5_5_to_scaffold raised',
                'pre_sanitize_smiles': None,
                'n_active_atoms': n_active, 'n_bonds_total': n_bonds_total,
                'atom_histogram': atom_hist_sym, 'fragment_histogram': dict(frag_hist),
                'cond_logP_norm': cond_i[0], 'cond_SAS_norm': cond_i[1],
            })
            continue

        # Try to build the molecule WITHOUT sanitize and capture what happens
        try:
            mol = RWMol()
            slot_to_rdkit = {}
            for j in range(len(aid)):
                if not am[j]: continue
                vid = int(aid[j])
                if vid == 0: continue
                elem, is_arom = _vocab_id_to_atom(vid)
                chg = _vocab_id_to_charge(vid)
                a = Atom(elem); a.SetIsAromatic(is_arom)
                if chg != 0: a.SetFormalCharge(chg)
                slot_to_rdkit[j] = mol.AddAtom(a)
            # Scaffold bonds
            for j in range(len(aid)):
                if j not in slot_to_rdkit: continue
                for k in range(j + 1, len(aid)):
                    if k not in slot_to_rdkit: continue
                    b_ = int(bc[j, k])
                    if b_ == 0: continue
                    mol.AddBond(slot_to_rdkit[j], slot_to_rdkit[k],
                                _BOND_CLASS_TO_TYPE[b_])
            # Terminal grafts (mirror the v5.5.1 valence-aware logic
            # so the dump reflects what assemble_molecule actually did)
            from meanflow.compose_full_molecule_zinc import (
                _current_explicit_valence, _max_valence_for, _BOND_ORDER,
            )
            for j in range(len(fid)):
                if not am[j]: continue
                f_ = int(fid[j])
                if f_ == 0: continue
                spec = _TERMINAL_SPECS.get(f_)
                if spec is None: continue
                host_idx = slot_to_rdkit[j]
                host = mol.GetAtomWithIdx(host_idx)
                attach_order = _BOND_ORDER.get(spec['attach'], 1.0)
                cur_val = _current_explicit_valence(mol, host_idx)
                max_val = _max_valence_for(host.GetSymbol(), host.GetFormalCharge())
                if cur_val + attach_order > max_val: continue
                if host.GetIsAromatic() and host.GetDegree() >= 3: continue
                frag_map = {}
                for k, (elem, is_arom, num_h) in enumerate(spec['atoms']):
                    a = Atom(elem); a.SetIsAromatic(is_arom)
                    if num_h > 0: a.SetNumExplicitHs(num_h)
                    frag_map[k] = mol.AddAtom(a)
                mol.AddBond(host_idx, frag_map[0], spec['attach'])
                for (a, b_, bt) in spec['bonds']:
                    mol.AddBond(frag_map[a], frag_map[b_], bt)

            # Try sanitize
            try:
                Chem.SanitizeMol(mol)
                # If we get here, the molecule is actually fine —
                # something else went wrong (shouldn't happen since this
                # mirrors assemble_molecule, but flag it).
                failure_records.append({
                    'idx': i, 'category': 'mismatch_sanitize_ok',
                    'error_class': None, 'error_msg': 'sanitize passed in dumper but failed in pipeline',
                    'pre_sanitize_smiles': Chem.MolToSmiles(mol, kekuleSmiles=False),
                    'n_active_atoms': n_active, 'n_bonds_total': n_bonds_total,
                    'atom_histogram': atom_hist_sym, 'fragment_histogram': dict(frag_hist),
                    'cond_logP_norm': cond_i[0], 'cond_SAS_norm': cond_i[1],
                })
            except Exception as san_e:
                err_class = type(san_e).__name__
                err_msg = str(san_e)
                # Bucket: pull the leading phrase (RDKit format is consistent)
                first_line = err_msg.split('\n')[0]
                bucket = first_line.split(':')[0][:60]
                if 'kekulize' in first_line.lower():
                    bucket = 'kekulize'
                elif 'explicit valence' in first_line.lower():
                    # Extract element from "atom # N X, V, is greater"
                    import re
                    m = re.search(r'atom # \d+ (\w+), (\d+),', first_line)
                    if m:
                        bucket = f'valence_{m.group(1)}_{m.group(2)}'
                    else:
                        bucket = 'valence_unknown'
                # Get pre-sanitize SMILES for inspection
                try:
                    pre_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False)
                except Exception:
                    pre_smi = '<unprintable>'
                failure_records.append({
                    'idx': i, 'category': 'sanitize_failed',
                    'sanitize_bucket': bucket,
                    'error_class': err_class, 'error_msg': first_line[:160],
                    'pre_sanitize_smiles': pre_smi[:200],
                    'n_active_atoms': n_active, 'n_bonds_total': n_bonds_total,
                    'atom_histogram': atom_hist_sym, 'fragment_histogram': dict(frag_hist),
                    'cond_logP_norm': cond_i[0], 'cond_SAS_norm': cond_i[1],
                })
        except Exception as build_e:
            # Pre-sanitize crash (shouldn't happen with v5.5.1 but possible)
            failure_records.append({
                'idx': i, 'category': 'pre_sanitize_crash',
                'error_class': type(build_e).__name__,
                'error_msg': str(build_e).split('\n')[0][:160],
                'pre_sanitize_smiles': None,
                'n_active_atoms': n_active, 'n_bonds_total': n_bonds_total,
                'atom_histogram': atom_hist_sym, 'fragment_histogram': dict(frag_hist),
                'cond_logP_norm': cond_i[0], 'cond_SAS_norm': cond_i[1],
            })

    # ─── Summary report ────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"FAILURE DUMP — {len(failure_records)} failures categorized")
    print(f"{'='*68}\n")

    # By category
    cat_counts = Counter(r['category'] for r in failure_records)
    print("Category breakdown:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:30s}  {count:4d}  ({100*count/n_failures:5.1f}%)")

    # By sanitize sub-bucket
    bucket_counts = Counter(
        r.get('sanitize_bucket', 'n/a') for r in failure_records
        if r['category'] == 'sanitize_failed'
    )
    if bucket_counts:
        print("\nSanitize sub-buckets:")
        for buck, count in bucket_counts.most_common():
            print(f"  {buck:30s}  {count:4d}")

    # Terminal class involvement: which terminal ids show up disproportionately
    # in failures? Compare frequency in failures vs frequency in all 1024.
    print("\nTerminal class involvement in failures (top 10):")
    fail_term = Counter()
    all_term  = Counter()
    for i in range(N_SAMPLES):
        for f_ in all_fragment_ids[i]:
            if f_ > 0: all_term[int(f_)] += 1
    for r in failure_records:
        for k, v in r['fragment_histogram'].items():
            fail_term[k] += v
    total_fail = sum(fail_term.values()) or 1
    total_all  = sum(all_term.values()) or 1
    print(f"  {'class':<6} {'in_fails':>10} {'in_all':>10} {'rel_freq':>10}")
    for cls, fc in fail_term.most_common(10):
        ac = all_term.get(cls, 0)
        fail_pct = fc / total_fail * 100
        all_pct  = ac / total_all * 100
        rel      = fail_pct / all_pct if all_pct > 0 else float('inf')
        marker = '  ⚠' if rel > 1.5 else ''
        print(f"  {cls:<6d} {fc:>10d} {ac:>10d} {rel:>9.2f}x{marker}")

    # Top 5 examples per sub-bucket
    print(f"\n{'='*68}")
    print("TOP 5 EXAMPLES PER SUB-BUCKET (pre-sanitize SMILES)")
    print(f"{'='*68}\n")
    by_bucket = defaultdict(list)
    for r in failure_records:
        key = r.get('sanitize_bucket', r['category'])
        by_bucket[key].append(r)

    for bucket, recs in sorted(by_bucket.items(), key=lambda kv: -len(kv[1])):
        print(f"\n── {bucket}  (n={len(recs)}) ──")
        for r in recs[:5]:
            smi = r.get('pre_sanitize_smiles') or '(no smiles)'
            errsnip = (r.get('error_msg') or '')[:70]
            print(f"  [{r['idx']:4d}] n_atoms={r['n_active_atoms']:2d}  "
                  f"frags={dict(list(r['fragment_histogram'].items())[:4])}")
            print(f"         {smi[:110]}")
            if errsnip and bucket != r.get('sanitize_bucket'):
                print(f"         err: {errsnip}")

    # ─── Save full JSON dump ───────────────────────────────────────────
    dump_path = f'{BASE}/failure_dump_v5_5_1.json'
    with open(dump_path, 'w') as f:
        json.dump({
            'n_samples_total': N_SAMPLES,
            'n_failures': n_failures,
            'failures': failure_records,
            'category_counts': dict(cat_counts),
            'sanitize_bucket_counts': dict(bucket_counts),
            'terminal_class_freq_in_failures': dict(fail_term),
            'terminal_class_freq_overall':    dict(all_term),
        }, f, indent=2, default=str)
    print(f"\n✓ Full dump saved to: {dump_path}")
    print(f"  ({n_failures} records, {os.path.getsize(dump_path)/1024:.0f} KB)")
