"""
HiMoFlow v5.4-ZINC — Conditional controllability evaluation.

Goal: beyond V·U·N (validity / uniqueness / novelty), measure whether
the model GENERATES MOLECULES THAT MATCH the requested (logP, SAS)
target. This is the eval that justifies the conditioning architecture:
if you ask for low-SAS easy-to-synthesize molecules, do you GET low-SAS
molecules?

Three diagnostics:

  1. Property prediction on generated SMILES
     - Compute logP via RDKit Crippen (matches ZINC250K's standard
       definition).
     - Compute SAS via SAScorer (Ertl & Schuffenhauer 2009, the
       standard SAS implementation).

  2. Target-vs-generated correlation
     - For a sweep of (logP_target, SAS_target) values across the
       training-distribution support, generate K samples per target,
       compute the mean property of generated molecules, and report
       Pearson r between target and mean-generated.
     - Strong conditioning → r close to 1.0.
     - Failed conditioning (model ignores condition) → r close to 0.0.

  3. Steering plot
     - 2D heatmap of (target_logP, target_SAS) vs (mean_gen_logP,
       mean_gen_SAS). The diagonal = perfect steering.

USAGE (in the train_and_eval notebook, after V·U·N section)
============================================================

    from controllability_eval_zinc import run_controllability_eval

    run_controllability_eval(
        a1=a1, a2=a2, stage2=stage2,
        train_labels=labels,
        cond_cols_resolved=cond_cols_resolved,
        # all the sampling hyperparameters from CONFIG
        a1_n_steps=A1_N_STEPS, a1_cfg_scale=A1_CFG_SCALE,
        a2_n_steps=A2_N_STEPS, a2_cfg_scale=A2_CFG_SCALE,
        a2_temperature=A2_TEMPERATURE,
        b5_n_steps=B5_N_STEPS, b5_temperature=B5_TEMPERATURE,
        b5_top_p=B5_TOP_P,
        n_per_target=64,
        device=device,
    )

Requires the SAScorer module from RDKit's contrib (it's bundled with
recent RDKit versions; if missing, install via:
    pip install rdkit-sascorer
or place sascorer.py from rdkit/Contrib/SA_Score in PYTHONPATH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
RDLogger.DisableLog("rdApp.*")

# Try the standard SAScorer location. Two common ways to access it.
try:
    from rdkit.Contrib.SA_Score import sascorer
    SAS_AVAILABLE = True
except ImportError:
    try:
        import sascorer
        SAS_AVAILABLE = True
    except ImportError:
        SAS_AVAILABLE = False
        print("WARNING: sascorer not found. Install via "
              "`pip install rdkit-sascorer` or place sascorer.py "
              "from rdkit/Contrib/SA_Score in PYTHONPATH. "
              "SAS controllability metrics will be skipped.")


def compute_logp(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    if m is None: return float('nan')
    try:
        return Crippen.MolLogP(m)
    except Exception:
        return float('nan')


def compute_sas(smi: str) -> float:
    if not SAS_AVAILABLE: return float('nan')
    m = Chem.MolFromSmiles(smi)
    if m is None: return float('nan')
    try:
        return sascorer.calculateScore(m)
    except Exception:
        return float('nan')


def denormalize(z: float, mean: float, std: float) -> float:
    return z * std + mean


def normalize(x: float, mean: float, std: float) -> float:
    return (x - mean) / std


def build_target_grid(train_logp_mean, train_logp_std,
                      train_sas_mean, train_sas_std,
                      n_logp=5, n_sas=5):
    """A grid of (logP, SAS) targets covering ±1.5σ of the training
    distribution. Returns list of (logp_target_z, sas_target_z) tuples
    in normalized space."""
    logp_grid = np.linspace(-1.5, 1.5, n_logp)  # in σ-units
    sas_grid = np.linspace(-1.5, 1.5, n_sas)
    targets = []
    for lz in logp_grid:
        for sz in sas_grid:
            targets.append((lz, sz))
    return targets


@torch.no_grad()
def sample_at_condition(
    a1, a2, stage2, condition_vec,
    n_samples: int,
    a1_n_steps: int, a1_cfg_scale: float,
    a2_n_steps: int, a2_cfg_scale: float, a2_temperature: float,
    b5_n_steps: int, b5_temperature: float, b5_top_p: float = 1.0,
    seed: int = 0,
    use_constraint_mask: bool = True,
    constraint_arom_atom_ids=(1, 5, 8, 9),
    constraint_forbid_classes=(7, 8, 9),
    device='cuda',
):
    """Draw n_samples molecules at a single fixed condition vector.

    condition_vec: 1-D tensor of normalized condition values (e.g.
        torch.tensor([logp_norm, sas_norm], device=device)).

    Returns: list[Optional[str]] of length n_samples (SMILES or None).
    """
    from meanflow.ring_layout_decoder import (
        aromatic_constraint_mask, build_bond_classes,
    )
    from meanflow.compose_full_molecule_zinc import assemble_to_smiles
    from meanflow.ring_atom_diffusion import M_MAX as A2_M_MAX

    cond_full = condition_vec.unsqueeze(0).expand(n_samples, -1).contiguous()
    a1_cond = cond_full[:, :a1.condition_dim].contiguous()
    a2_cond = cond_full[:, :a2.condition_dim].contiguous()
    s2_cond = cond_full[:, :stage2.condition_dim].contiguous()

    layouts = a1.sample(condition=a1_cond, n_steps=a1_n_steps,
                        seed=seed, cfg_scale=a1_cfg_scale)

    bc_list, am_list, ar_list, valid_idx = [], [], [], []
    for b in range(n_samples):
        R    = layouts['R'][b].cpu().numpy()
        F_   = layouts['F'][b].cpu().numpy()
        L_   = layouts['L'][b].cpu().numpy()
        Plen = layouts['P_len'][b].cpu().numpy()
        Ppos = layouts['P_pos'][b].cpu().numpy()
        try:
            bc, am = build_bond_classes(R, F_, L_, Plen, Ppos,
                                         M_MAX_out=A2_M_MAX)
            ar = aromatic_constraint_mask(R, F_, L_, Plen, Ppos,
                                           M_MAX_out=A2_M_MAX)
            bc_list.append(bc); am_list.append(am); ar_list.append(ar)
            valid_idx.append(b)
        except Exception:
            pass

    if not valid_idx:
        return [None] * n_samples

    bc_t = torch.from_numpy(np.stack(bc_list)).long().to(device)
    am_t = torch.from_numpy(np.stack(am_list)).bool().to(device)
    ar_t = torch.from_numpy(np.stack(ar_list)).bool().to(device)
    a2_cond_d = a2_cond[valid_idx]
    s2_cond_d = s2_cond[valid_idx]

    atom_ids = a2.sample(
        bond_classes=bc_t, atom_mask=am_t, arom_mask=ar_t,
        condition=a2_cond_d,
        n_steps=a2_n_steps, seed=seed,
        cfg_scale=a2_cfg_scale, temperature=a2_temperature,
    )

    # Stage 2 sampling with constraint mask
    arom_atom_ids = torch.tensor(constraint_arom_atom_ids, device=device)
    is_arom = torch.isin(atom_ids, arom_atom_ids)
    forbid_classes = torch.tensor(constraint_forbid_classes, device=device)
    orig_forward = stage2.forward

    def hooked_forward(*args, **kwargs):
        logits = orig_forward(*args, **kwargs)
        if use_constraint_mask:
            mask = is_arom[:logits.shape[0]].unsqueeze(-1)
            for cls_idx in forbid_classes:
                logits[..., cls_idx] = torch.where(
                    mask.squeeze(-1),
                    torch.full_like(logits[..., cls_idx], float('-inf')),
                    logits[..., cls_idx],
                )
        return logits

    stage2.forward = hooked_forward
    try:
        fragment_ids = stage2.sample(
            atom_ids=atom_ids, bond_classes=bc_t, atom_mask=am_t,
            condition=s2_cond_d,
            n_steps=b5_n_steps, seed=seed,
            temperature=b5_temperature,
        )
    finally:
        stage2.forward = orig_forward

    # Assemble each
    smis = []
    for b in range(len(valid_idx)):
        smi = assemble_to_smiles(
            atom_ids[b].cpu().numpy(),
            bc_t[b].cpu().numpy(),
            am_t[b].cpu().numpy(),
            fragment_ids[b].cpu().numpy(),
        )
        smis.append(smi)

    # Pad to n_samples with None for invalid layouts
    full = [None] * n_samples
    for i, b in enumerate(valid_idx):
        full[b] = smis[i]
    return full


def run_controllability_eval(
    a1, a2, stage2,
    train_labels,
    cond_cols_resolved,
    n_per_target: int = 64,
    n_logp_grid: int = 5,
    n_sas_grid: int = 5,
    a1_n_steps=20, a1_cfg_scale=2.0,
    a2_n_steps=20, a2_cfg_scale=1.5, a2_temperature=1.0,
    b5_n_steps=16, b5_temperature=1.3, b5_top_p=1.0,
    seed=42, device='cuda',
):
    """Sweep a (logP_target, SAS_target) grid; for each target, sample
    n_per_target molecules and compute their property statistics.
    Returns a pandas DataFrame and prints summary stats.

    Assumes the conditioning vector is (logP_norm, SAS_norm) in that
    order. If your training used a different order, the column order
    of cond_cols_resolved should be inspected.
    """
    # Recover normalization stats from training labels: condition[i]
    # holds the z-scored values, so we need the original mean/std.
    # We stored the augmented CSV so we can read it from there. But
    # easier: in the training labels, each item has 'condition' which
    # is normalized. The TRAIN/UN-NORMALIZED logP and SAS are not in
    # the labels, so we need to load them from the augmented CSV.
    raise NotImplementedError(
        "This is the eval skeleton. To wire it up, you need to "
        "pass the augmented CSV path (or the (logp_mean, logp_std, "
        "sas_mean, sas_std) tuple) so that `denormalize` can convert "
        "between target z-scores and physical (logP, SAS) values "
        "for the controllability heatmap. See the notebook integration "
        "stub at the bottom of this file."
    )


# ────────────────────────────────────────────────────────────────────
# NOTEBOOK INTEGRATION STUB — paste at end of train_and_eval notebook
# ────────────────────────────────────────────────────────────────────
NOTEBOOK_INTEGRATION_STUB = r"""
# ═══════════════════════════════════════════════════════════════════
#  Section 12 — SAS/logP controllability eval
# ═══════════════════════════════════════════════════════════════════
import sys; sys.path.insert(0, BASE)
from controllability_eval_zinc import (
    sample_at_condition, compute_logp, compute_sas, SAS_AVAILABLE,
)
import matplotlib.pyplot as plt

# Pull normalization stats from the augmented CSV
df_aug = pd.read_csv(AUGMENTED_CSV)
LOGP_MEAN = df_aug['logP'].mean(); LOGP_STD = df_aug['logP'].std()
SAS_MEAN  = df_aug['SAS'].mean();  SAS_STD  = df_aug['SAS'].std()
print(f'logP: μ={LOGP_MEAN:.3f}  σ={LOGP_STD:.3f}')
print(f'SAS:  μ={SAS_MEAN:.3f}   σ={SAS_STD:.3f}')

# Target grid: ±1.5σ in each axis, 5×5
N_GRID = 5
SIGMA_RANGE = 1.5
N_PER_TARGET = 64
logp_zs = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, N_GRID)
sas_zs  = np.linspace(-SIGMA_RANGE, SIGMA_RANGE, N_GRID)

results = []
for li, lz in enumerate(logp_zs):
    for si, sz in enumerate(sas_zs):
        cond_vec = torch.tensor([lz, sz], device=device, dtype=torch.float32)
        smis = sample_at_condition(
            a1, a2, stage2, cond_vec, n_samples=N_PER_TARGET,
            a1_n_steps=A1_N_STEPS, a1_cfg_scale=A1_CFG_SCALE,
            a2_n_steps=A2_N_STEPS, a2_cfg_scale=A2_CFG_SCALE,
            a2_temperature=A2_TEMPERATURE,
            b5_n_steps=B5_N_STEPS, b5_temperature=B5_TEMPERATURE,
            b5_top_p=B5_TOP_P,
            seed=SAMPLE_SEED + li * N_GRID + si,
            use_constraint_mask=USE_CONSTRAINT_MASK,
            constraint_arom_atom_ids=CONSTRAINT_AROM_ATOM_IDS,
            constraint_forbid_classes=CONSTRAINT_FORBID_CLASSES,
            device=device,
        )
        valid_smis = [s for s in smis if s is not None]
        logps = [compute_logp(s) for s in valid_smis]
        logps = [v for v in logps if not np.isnan(v)]
        if SAS_AVAILABLE:
            sass = [compute_sas(s) for s in valid_smis]
            sass = [v for v in sass if not np.isnan(v)]
        else:
            sass = []
        results.append({
            'lz': lz, 'sz': sz,
            'logP_target': lz * LOGP_STD + LOGP_MEAN,
            'SAS_target':  sz * SAS_STD + SAS_MEAN,
            'logP_gen_mean': float(np.mean(logps)) if logps else float('nan'),
            'logP_gen_std':  float(np.std(logps))  if logps else float('nan'),
            'SAS_gen_mean':  float(np.mean(sass))  if sass  else float('nan'),
            'SAS_gen_std':   float(np.std(sass))   if sass  else float('nan'),
            'n_valid': len(valid_smis),
        })
        print(f'  [{li},{si}] target=(logP={lz*LOGP_STD+LOGP_MEAN:.2f}, '
              f'SAS={sz*SAS_STD+SAS_MEAN:.2f}) → '
              f'gen=(logP={np.mean(logps) if logps else float("nan"):.2f}, '
              f'SAS={np.mean(sass) if sass else float("nan"):.2f}), '
              f'n_valid={len(valid_smis)}')

df_ctrl = pd.DataFrame(results)

# Pearson r — controllability metric
r_logp = df_ctrl[['logP_target', 'logP_gen_mean']].corr().iloc[0,1]
r_sas  = df_ctrl[['SAS_target',  'SAS_gen_mean']].corr().iloc[0,1] \
         if SAS_AVAILABLE else float('nan')
print('\n' + '=' * 60)
print(f'  Controllability r(target, generated):')
print(f'    logP: r = {r_logp:.3f}')
if SAS_AVAILABLE: print(f'    SAS:  r = {r_sas:.3f}')
print(f'  (1.0 = perfect steering; 0.0 = no response to condition)')
print('=' * 60)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax = axes[0]
ax.scatter(df_ctrl['logP_target'], df_ctrl['logP_gen_mean'])
lo, hi = df_ctrl['logP_target'].min(), df_ctrl['logP_target'].max()
ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x (perfect)')
ax.set_xlabel('logP target'); ax.set_ylabel('logP generated (mean)')
ax.set_title(f'logP controllability (r = {r_logp:.3f})')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
if SAS_AVAILABLE:
    ax.scatter(df_ctrl['SAS_target'], df_ctrl['SAS_gen_mean'])
    lo, hi = df_ctrl['SAS_target'].min(), df_ctrl['SAS_target'].max()
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4)
    ax.set_xlabel('SAS target'); ax.set_ylabel('SAS generated (mean)')
    ax.set_title(f'SAS controllability (r = {r_sas:.3f})')
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, 'sascorer unavailable', ha='center', va='center')
    ax.axis('off')
plt.tight_layout(); plt.show()
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n\n=== NOTEBOOK INTEGRATION STUB ===\n")
    print(NOTEBOOK_INTEGRATION_STUB)
