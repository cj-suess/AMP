#!/usr/bin/env python
# coding: utf-8

# # SK Model: Fair Comparative Benchmark
# Three experiments benchmarking GD, SGD, AMP, and Spectral on the SK model.
#
# Experiment 1 — Single-run quality averaged over seeds.
#   Each algorithm runs exactly once per GOE instance. Results are averaged
#   over NUM_SEEDS instances. No algorithm gets more attempts than another.
#   This measures intrinsic per-run quality.
#
# Experiment 2 — Fixed wall-clock budget.
#   Each algorithm is given TIME_BUDGET_SEC seconds per (N, iter) cell.
#   It runs as many restarts as it can fit in that budget and reports the
#   best result found. This measures practical value per unit of real time.
#
# Experiment 3 — AMP vs Spectral: restart sweep (optimality-seeking).
#   GD and SGD are dropped. AMP and Spectral are each given a fixed restart
#   budget [1, 10, 25, 50, 100] on a focused set of N values [500, 1000, 2000, 5000].
#   AMP uses 500 iterations per restart (good convergence without being wasteful).
#   This answers: how much does AMP's quality advantage grow as you spend more,
#   and what is the FLOP cost of that advantage relative to Spectral?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from scipy.sparse.linalg import eigsh

warnings.filterwarnings('ignore')

print('Libraries loaded.')


# =============================================================================
# 1. Core Algorithms
# =============================================================================

def generate_sk_matrix(N, rng):
    J = rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N))
    J = (J + J.T) / np.sqrt(2)
    np.fill_diagonal(J, 0.0)
    return J

def calculate_energy(sigma, J):
    return -0.5 * (sigma @ J @ sigma)

def energy_per_spin(sigma, J):
    return calculate_energy(sigma, J) / len(sigma)

# ── Gradient Descent ─────────────────────────────────────────────────────────

def gradient_descent_sk(J, num_iterations, learning_rate, convergence_tol, rng):
    N = J.shape[0]
    sigma = rng.uniform(-1.0, 1.0, size=N)
    energy_history = []
    convergence_iter = num_iterations
    converged = False
    for i in range(num_iterations):
        gradient = -(J @ sigma)
        sigma = sigma - learning_rate * gradient
        sigma = np.clip(sigma, -1.0, 1.0)
        e = calculate_energy(sigma, J)
        energy_history.append(float(e))
        if i > 0 and not converged:
            delta = abs(energy_history[-1] - energy_history[-2])
            if delta < convergence_tol * abs(energy_history[-1] + 1e-12):
                convergence_iter = i + 1
                converged = True
    return sigma, energy_history, convergence_iter

# ── SGD ──────────────────────────────────────────────────────────────────────

def stochastic_gradient_descent_sk(J, num_iterations, learning_rate,
                                   batch_size, eval_every, convergence_tol, rng):
    N = J.shape[0]
    batch_size = min(batch_size, N)
    sigma = rng.uniform(-1.0, 1.0, size=N)
    energy_history, eval_steps = [], []
    convergence_iter = num_iterations
    converged = False
    for i in range(num_iterations):
        cols = rng.choice(N, size=batch_size, replace=False)
        grad_estimate = -(N / batch_size) * (J[:, cols] @ sigma[cols])
        sigma = sigma - learning_rate * grad_estimate
        sigma = np.clip(sigma, -1.0, 1.0)
        should_eval = ((i + 1) % eval_every == 0) or (i == num_iterations - 1)
        if should_eval:
            e = calculate_energy(sigma, J)
            energy_history.append(float(e))
            eval_steps.append(i + 1)
            if len(energy_history) > 1 and not converged:
                delta = abs(energy_history[-1] - energy_history[-2])
                if delta < convergence_tol * abs(energy_history[-1] + 1e-12):
                    convergence_iter = i + 1
                    converged = True
    return sigma, energy_history, eval_steps, convergence_iter

# ── AMP ──────────────────────────────────────────────────────────────────────

def amp_sk(J, m_init, num_iterations, damping):
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    betas = np.linspace(0.1, 2.5, num_iterations)
    for beta in betas:
        onsager = np.mean(1.0 - m**2)
        h_target = J @ m - beta * onsager * m_old
        h = damping * h + (1.0 - damping) * h_target
        m_old = np.copy(m)
        m = np.tanh(beta * h)
    return np.sign(m)

def greedy_quench(sigma, J):
    sigma_opt = np.copy(sigma).astype(float)
    improved = True
    passes = 0
    while improved:
        improved = False
        local_fields = J @ sigma_opt
        frustration = sigma_opt * local_fields
        idx = np.argmin(frustration)
        if frustration[idx] < 0:
            sigma_opt[idx] *= -1
            improved = True
        passes += 1
    return sigma_opt, passes

def get_orthogonal_start(N, rng, scale=0.001):
    v = rng.normal(size=N)
    v /= np.linalg.norm(v)
    return v * scale

# ── Spectral ─────────────────────────────────────────────────────────────────

def spectral_sk(J, refine=True):
    N = J.shape[0]
    eigenvalues, eigenvectors = eigsh(
        J, k=1, which='LA', return_eigenvectors=True,
        maxiter=1000, tol=1e-6,
    )
    v = eigenvectors[:, 0]
    spins = np.sign(v)
    spins[spins == 0] = 1.0
    quench_passes = 0
    if refine:
        spins, quench_passes = greedy_quench(spins, J)
    info = {
        'top_eigenvalue': float(eigenvalues[0]),
        'lanczos_iters_estimate': max(20, int(np.sqrt(N))),
        'quench_passes': quench_passes,
        'eigenvector': v,
    }
    return spins, info

print('Algorithm definitions ready.')


# =============================================================================
# 2. FLOP Estimators
# =============================================================================

def flops_gd(N, iters):
    return iters * 2 * (2 * N * N)

def flops_sgd(N, iters, batch_size, num_energy_evals):
    return (iters * (2 * N * batch_size + 3 * N)
            + num_energy_evals * 2 * N * N)

def flops_amp_single(N, amp_iters, quench_passes):
    return amp_iters * (2 * N * N) + quench_passes * (2 * N * N)

def flops_spectral(N, lanczos_iters, quench_passes):
    return (lanczos_iters + quench_passes + 1) * (2 * N * N)

print('FLOP estimators defined.')


# =============================================================================
# 3. Configuration
# =============================================================================

# Experiments 1 & 2
N_VALUES         = [50, 100, 250, 500, 1000, 2000, 3000, 5000]
ITERATION_VALUES = [100, 250, 500, 1000]
NUM_SEEDS        = 10
TIME_BUDGET_SEC  = 2.0

# Experiment 3
EXP3_N_VALUES      = [500, 1000, 2000, 5000]
EXP3_RESTART_VALUES = [1, 10, 25, 50, 100]
EXP3_AMP_ITERS     = 500    # fixed iteration count per AMP restart
EXP3_NUM_SEEDS     = 5      # seeds to average over per (N, restarts) cell
PARISI_VALUE       = -0.7633

# Shared hyperparameters
GD_LR           = 0.1
GD_CONV_TOL     = 1e-5
SGD_LR          = 0.05
SGD_BATCH_SIZE  = 64
SGD_EVAL_EVERY  = 10
AMP_DAMPING     = 0.7
SPECTRAL_REFINE = True

ALGO_COLORS  = {'GD': '#E91E63', 'SGD': '#7E57C2', 'AMP': '#00BCD4', 'SPEC': '#FF9800'}
ITER_COLORS  = {100: '#FF6B6B', 250: '#4ECDC4', 500: '#45B7D1', 1000: '#2E86AB'}
ITER_MARKERS = {100: 'o', 250: 's', 500: '^', 1000: 'D'}
N_COLORS     = {500: '#E91E63', 1000: '#7E57C2', 2000: '#00BCD4', 5000: '#FF9800'}
N_MARKERS    = {500: 'o', 1000: 's', 2000: '^', 5000: 'D'}

print(f'N values (Exp 1&2):      {N_VALUES}')
print(f'Iteration values:        {ITERATION_VALUES}')
print(f'Seeds (Exp 1):           {NUM_SEEDS}')
print(f'Time budget (Exp 2):     {TIME_BUDGET_SEC}s per cell')
print(f'N values (Exp 3):        {EXP3_N_VALUES}')
print(f'Restart sweep (Exp 3):   {EXP3_RESTART_VALUES}')
print(f'AMP iters per restart:   {EXP3_AMP_ITERS}')
print(f'Seeds (Exp 3):           {EXP3_NUM_SEEDS}')


# =============================================================================
# 4. Experiment 1 — Single-run quality, averaged over seeds
# =============================================================================

print('\n' + '='*70)
print('EXPERIMENT 1: Single-run quality averaged over seeds')
print('='*70)

exp1_records = []
total_cells = len(ITERATION_VALUES) * len(N_VALUES)
cell_idx = 0

for ITER in ITERATION_VALUES:
    for N in N_VALUES:
        cell_idx += 1
        theoretical_limit = PARISI_VALUE * N

        gd_energies, sgd_energies, amp_energies, spec_energies = [], [], [], []
        gd_walls,    sgd_walls,    amp_walls,    spec_walls    = [], [], [], []
        gd_flops_list, sgd_flops_list, amp_flops_list, spec_flops_list = [], [], [], []
        gd_conv_iters = []

        for seed in range(NUM_SEEDS):
            rng = np.random.default_rng(seed)
            J = generate_sk_matrix(N, rng)

            t0 = time.perf_counter()
            _, gd_curve, gd_conv_iter = gradient_descent_sk(
                J, ITER, GD_LR, GD_CONV_TOL, rng)
            gd_walls.append(time.perf_counter() - t0)
            gd_energies.append(gd_curve[-1])
            gd_flops_list.append(flops_gd(N, ITER))
            gd_conv_iters.append(gd_conv_iter)

            t0 = time.perf_counter()
            _, sgd_curve, _, sgd_conv_iter = stochastic_gradient_descent_sk(
                J, ITER, SGD_LR, SGD_BATCH_SIZE, SGD_EVAL_EVERY, GD_CONV_TOL, rng)
            sgd_walls.append(time.perf_counter() - t0)
            sgd_energies.append(sgd_curve[-1])
            sgd_flops_list.append(flops_sgd(N, ITER, min(SGD_BATCH_SIZE, N), len(sgd_curve)))

            m_init = get_orthogonal_start(N, rng)
            t0 = time.perf_counter()
            raw = amp_sk(J, m_init, ITER, AMP_DAMPING)
            quenched, qp = greedy_quench(raw, J)
            amp_walls.append(time.perf_counter() - t0)
            amp_energies.append(float(calculate_energy(quenched, J)))
            amp_flops_list.append(flops_amp_single(N, ITER, qp))

            t0 = time.perf_counter()
            spec_spins, spec_info = spectral_sk(J, refine=SPECTRAL_REFINE)
            spec_walls.append(time.perf_counter() - t0)
            spec_energies.append(float(calculate_energy(spec_spins, J)))
            spec_flops_list.append(flops_spectral(
                N, spec_info['lanczos_iters_estimate'], spec_info['quench_passes']))

        def stats(vals):
            return np.mean(vals), np.std(vals)

        gd_e_mean,   gd_e_std   = stats(gd_energies)
        sgd_e_mean,  sgd_e_std  = stats(sgd_energies)
        amp_e_mean,  amp_e_std  = stats(amp_energies)
        spec_e_mean, spec_e_std = stats(spec_energies)

        def gap_pct(e): return 100 * abs(e - theoretical_limit) / abs(theoretical_limit)

        print(f'[{cell_idx:02d}/{total_cells}]  N={N:>5d}  iter={ITER:<5d}  '
              f'GD: {gd_e_mean/N:+.4f}  '
              f'SGD: {sgd_e_mean/N:+.4f}  '
              f'AMP: {amp_e_mean/N:+.4f}  '
              f'SPEC: {spec_e_mean/N:+.4f}  '
              f'PARISI: {PARISI_VALUE:+.4f}')

        exp1_records.append(dict(
            experiment        = 1,
            iterations        = ITER,
            N                 = N,
            theoretical_limit = round(theoretical_limit, 4),
            parisi_value      = PARISI_VALUE,
            gd_mean_eN        = round(gd_e_mean / N, 5),
            sgd_mean_eN       = round(sgd_e_mean / N, 5),
            amp_mean_eN       = round(amp_e_mean / N, 5),
            spec_mean_eN      = round(spec_e_mean / N, 5),
            gd_std_eN         = round(gd_e_std / N, 5),
            sgd_std_eN        = round(sgd_e_std / N, 5),
            amp_std_eN        = round(amp_e_std / N, 5),
            spec_std_eN       = round(spec_e_std / N, 5),
            gd_gap_pct        = round(gap_pct(gd_e_mean), 3),
            sgd_gap_pct       = round(gap_pct(sgd_e_mean), 3),
            amp_gap_pct       = round(gap_pct(amp_e_mean), 3),
            spec_gap_pct      = round(gap_pct(spec_e_mean), 3),
            gd_wall_sec       = round(np.mean(gd_walls), 5),
            sgd_wall_sec      = round(np.mean(sgd_walls), 5),
            amp_wall_sec      = round(np.mean(amp_walls), 5),
            spec_wall_sec     = round(np.mean(spec_walls), 5),
            gd_flops          = int(np.mean(gd_flops_list)),
            sgd_flops         = int(np.mean(sgd_flops_list)),
            amp_flops         = int(np.mean(amp_flops_list)),
            spec_flops        = int(np.mean(spec_flops_list)),
            winner            = min(
                [('GD', gd_e_mean), ('SGD', sgd_e_mean),
                 ('AMP', amp_e_mean), ('SPEC', spec_e_mean)],
                key=lambda x: x[1])[0],
            num_seeds         = NUM_SEEDS,
        ))

df1 = pd.DataFrame(exp1_records)
print(f'\n✓ Experiment 1 complete. {len(df1)} records.')


# =============================================================================
# 5. Experiment 2 — Fixed wall-clock budget
# =============================================================================

print('\n' + '='*70)
print(f'EXPERIMENT 2: Fixed wall-clock budget ({TIME_BUDGET_SEC}s per cell)')
print('='*70)

exp2_records = []
cell_idx = 0
FIXED_SEED = 42

for ITER in ITERATION_VALUES:
    for N in N_VALUES:
        cell_idx += 1
        theoretical_limit = PARISI_VALUE * N
        rng = np.random.default_rng(FIXED_SEED)
        J = generate_sk_matrix(N, rng)

        def gap_pct(e): return 100 * abs(e - theoretical_limit) / abs(theoretical_limit)

        gd_best = np.inf; gd_restarts = 0; gd_total_flops = 0
        t_gd_start = time.perf_counter()
        while time.perf_counter() - t_gd_start < TIME_BUDGET_SEC:
            _, gd_curve, _ = gradient_descent_sk(J, ITER, GD_LR, GD_CONV_TOL, rng)
            e = gd_curve[-1]
            if e < gd_best: gd_best = e
            gd_total_flops += flops_gd(N, ITER)
            gd_restarts += 1
        gd_wall = time.perf_counter() - t_gd_start

        sgd_best = np.inf; sgd_restarts = 0; sgd_total_flops = 0
        t_sgd_start = time.perf_counter()
        while time.perf_counter() - t_sgd_start < TIME_BUDGET_SEC:
            _, sgd_curve, _, _ = stochastic_gradient_descent_sk(
                J, ITER, SGD_LR, SGD_BATCH_SIZE, SGD_EVAL_EVERY, GD_CONV_TOL, rng)
            e = sgd_curve[-1]
            if e < sgd_best: sgd_best = e
            sgd_total_flops += flops_sgd(N, ITER, min(SGD_BATCH_SIZE, N), len(sgd_curve))
            sgd_restarts += 1
        sgd_wall = time.perf_counter() - t_sgd_start

        amp_best = np.inf; amp_restarts = 0; amp_total_flops = 0
        t_amp_start = time.perf_counter()
        while time.perf_counter() - t_amp_start < TIME_BUDGET_SEC:
            m_init = get_orthogonal_start(N, rng)
            raw = amp_sk(J, m_init, ITER, AMP_DAMPING)
            quenched, qp = greedy_quench(raw, J)
            e = float(calculate_energy(quenched, J))
            if e < amp_best: amp_best = e
            amp_total_flops += flops_amp_single(N, ITER, qp)
            amp_restarts += 1
        amp_wall = time.perf_counter() - t_amp_start

        eigenvalues, eigenvectors = eigsh(
            J, k=1, which='LA', return_eigenvectors=True, maxiter=1000, tol=1e-6)
        v = eigenvectors[:, 0]
        spec_best = np.inf; spec_restarts = 0
        spec_total_flops = flops_spectral(N, max(20, int(np.sqrt(N))), 0)
        t_spec_start = time.perf_counter()
        while time.perf_counter() - t_spec_start < TIME_BUDGET_SEC:
            if spec_restarts == 0:
                spins = np.sign(v)
            else:
                flip_frac = rng.uniform(0.05, 0.3)
                flip_mask = rng.random(N) < flip_frac
                spins = np.sign(v)
                spins[flip_mask] *= -1
            spins[spins == 0] = 1.0
            spins, qp = greedy_quench(spins, J)
            e = float(calculate_energy(spins, J))
            if e < spec_best: spec_best = e
            spec_total_flops += qp * (2 * N * N)
            spec_restarts += 1
        spec_wall = time.perf_counter() - t_spec_start

        print(f'[{cell_idx:02d}/{total_cells}]  N={N:>5d}  iter={ITER:<5d}  '
              f'GD: {gd_best/N:+.4f} (x{gd_restarts})  '
              f'SGD: {sgd_best/N:+.4f} (x{sgd_restarts})  '
              f'AMP: {amp_best/N:+.4f} (x{amp_restarts})  '
              f'SPEC: {spec_best/N:+.4f} (x{spec_restarts})')

        exp2_records.append(dict(
            experiment        = 2,
            iterations        = ITER,
            N                 = N,
            time_budget_sec   = TIME_BUDGET_SEC,
            theoretical_limit = round(theoretical_limit, 4),
            parisi_value      = PARISI_VALUE,
            gd_best_eN        = round(gd_best / N, 5),
            sgd_best_eN       = round(sgd_best / N, 5),
            amp_best_eN       = round(amp_best / N, 5),
            spec_best_eN      = round(spec_best / N, 5),
            gd_gap_pct        = round(gap_pct(gd_best), 3),
            sgd_gap_pct       = round(gap_pct(sgd_best), 3),
            amp_gap_pct       = round(gap_pct(amp_best), 3),
            spec_gap_pct      = round(gap_pct(spec_best), 3),
            gd_restarts       = gd_restarts,
            sgd_restarts      = sgd_restarts,
            amp_restarts      = amp_restarts,
            spec_restarts     = spec_restarts,
            gd_wall_sec       = round(gd_wall, 3),
            sgd_wall_sec      = round(sgd_wall, 3),
            amp_wall_sec      = round(amp_wall, 3),
            spec_wall_sec     = round(spec_wall, 3),
            gd_flops          = gd_total_flops,
            sgd_flops         = sgd_total_flops,
            amp_flops         = amp_total_flops,
            spec_flops        = spec_total_flops,
            winner            = min(
                [('GD', gd_best), ('SGD', sgd_best),
                 ('AMP', amp_best), ('SPEC', spec_best)],
                key=lambda x: x[1])[0],
        ))

df2 = pd.DataFrame(exp2_records)
print(f'\n✓ Experiment 2 complete. {len(df2)} records.')


# =============================================================================
# 6. Experiment 3 — AMP vs Spectral: restart sweep (optimality-seeking)
#
#   GD and SGD are dropped — we already know they underperform.
#   Both AMP and Spectral are given exactly R restarts from the sweep
#   EXP3_RESTART_VALUES = [1, 10, 25, 50, 100].
#   AMP uses EXP3_AMP_ITERS iterations per restart.
#   Spectral uses the same sign-flip restart strategy as Experiment 2.
#   Results are averaged over EXP3_NUM_SEEDS independent GOE matrices.
#
#   Key outputs per (N, restarts) cell:
#     - Mean best energy/N found by AMP and Spectral
#     - Total wall time and FLOPs for each (summed over all restarts)
#     - FLOP ratio AMP/Spectral at each restart count
#     - The gap between AMP and Spectral quality as restarts increase
# =============================================================================

print('\n' + '='*70)
print('EXPERIMENT 3: AMP vs Spectral — Restart Budget Sweep')
print(f'  N values:        {EXP3_N_VALUES}')
print(f'  Restart values:  {EXP3_RESTART_VALUES}')
print(f'  AMP iters/run:   {EXP3_AMP_ITERS}')
print(f'  Seeds:           {EXP3_NUM_SEEDS}')
print('='*70)

exp3_records = []
total_cells_3 = len(EXP3_N_VALUES) * len(EXP3_RESTART_VALUES)
cell_idx = 0

for N in EXP3_N_VALUES:
    theoretical_limit = PARISI_VALUE * N

    for num_restarts in EXP3_RESTART_VALUES:
        cell_idx += 1

        amp_best_list  = []
        spec_best_list = []
        amp_wall_list  = []
        spec_wall_list = []
        amp_flops_list = []
        spec_flops_list= []

        for seed in range(EXP3_NUM_SEEDS):
            rng = np.random.default_rng(seed * 1000 + N)
            J = generate_sk_matrix(N, rng)

            # ── AMP: exactly num_restarts runs, report best ───────────────────
            amp_best = np.inf
            amp_total_flops = 0
            t0 = time.perf_counter()
            for _ in range(num_restarts):
                m_init = get_orthogonal_start(N, rng)
                raw = amp_sk(J, m_init, EXP3_AMP_ITERS, AMP_DAMPING)
                quenched, qp = greedy_quench(raw, J)
                e = float(calculate_energy(quenched, J))
                if e < amp_best:
                    amp_best = e
                amp_total_flops += flops_amp_single(N, EXP3_AMP_ITERS, qp)
            amp_wall = time.perf_counter() - t0

            # ── Spectral: eigenvector computed once, then num_restarts quenches ─
            # Restart 0: standard sign rounding of eigenvector.
            # Restarts 1+: random sign-flip fraction before quenching.
            t_eig = time.perf_counter()
            eigenvalues, eigenvectors = eigsh(
                J, k=1, which='LA', return_eigenvectors=True,
                maxiter=1000, tol=1e-6)
            eig_time = time.perf_counter() - t_eig
            v = eigenvectors[:, 0]

            spec_best = np.inf
            spec_total_flops = flops_spectral(N, max(20, int(np.sqrt(N))), 0)
            t0 = time.perf_counter()
            for r in range(num_restarts):
                if r == 0:
                    spins = np.sign(v)
                else:
                    flip_frac = rng.uniform(0.05, 0.30)
                    flip_mask = rng.random(N) < flip_frac
                    spins = np.sign(v)
                    spins[flip_mask] *= -1
                spins[spins == 0] = 1.0
                spins, qp = greedy_quench(spins, J)
                e = float(calculate_energy(spins, J))
                if e < spec_best:
                    spec_best = e
                spec_total_flops += qp * (2 * N * N)
            spec_wall = (time.perf_counter() - t0) + eig_time

            amp_best_list.append(amp_best)
            spec_best_list.append(spec_best)
            amp_wall_list.append(amp_wall)
            spec_wall_list.append(spec_wall)
            amp_flops_list.append(amp_total_flops)
            spec_flops_list.append(spec_total_flops)

        amp_mean_eN  = np.mean(amp_best_list)  / N
        spec_mean_eN = np.mean(spec_best_list) / N
        amp_gap      = 100 * abs(np.mean(amp_best_list)  - theoretical_limit) / abs(theoretical_limit)
        spec_gap     = 100 * abs(np.mean(spec_best_list) - theoretical_limit) / abs(theoretical_limit)
        amp_wall_mean  = np.mean(amp_wall_list)
        spec_wall_mean = np.mean(spec_wall_list)
        amp_flops_mean  = int(np.mean(amp_flops_list))
        spec_flops_mean = int(np.mean(spec_flops_list))
        flop_ratio = amp_flops_mean / max(spec_flops_mean, 1)
        quality_gap_pp = amp_gap - spec_gap   # positive = AMP is closer to Parisi

        winner = 'AMP' if amp_mean_eN < spec_mean_eN else 'SPEC'

        print(f'[{cell_idx:02d}/{total_cells_3}]  N={N:>5d}  restarts={num_restarts:<4d}  '
              f'AMP: {amp_mean_eN:+.4f}  '
              f'SPEC: {spec_mean_eN:+.4f}  '
              f'PARISI: {PARISI_VALUE:+.4f}  '
              f'| AMP_gap: {amp_gap:.2f}%  SPEC_gap: {spec_gap:.2f}%  '
              f'| AMP_t: {amp_wall_mean:.2f}s  SPEC_t: {spec_wall_mean:.3f}s  '
              f'| FLOP_ratio: {flop_ratio:.1f}x')

        exp3_records.append(dict(
            experiment        = 3,
            N                 = N,
            num_restarts      = num_restarts,
            amp_iters_per_run = EXP3_AMP_ITERS,
            theoretical_limit = round(theoretical_limit, 4),
            parisi_value      = PARISI_VALUE,
            # Quality
            amp_mean_eN       = round(amp_mean_eN, 5),
            spec_mean_eN      = round(spec_mean_eN, 5),
            amp_gap_pct       = round(amp_gap, 3),
            spec_gap_pct      = round(spec_gap, 3),
            quality_gap_pp    = round(quality_gap_pp, 3),  # AMP advantage in pp (negative = SPEC wins)
            # Timing
            amp_wall_sec      = round(amp_wall_mean, 4),
            spec_wall_sec     = round(spec_wall_mean, 4),
            wall_ratio        = round(amp_wall_mean / max(spec_wall_mean, 1e-9), 2),
            # FLOPs
            amp_flops         = amp_flops_mean,
            spec_flops        = spec_flops_mean,
            flop_ratio        = round(flop_ratio, 1),
            winner            = winner,
            num_seeds         = EXP3_NUM_SEEDS,
        ))

df3 = pd.DataFrame(exp3_records)
print(f'\n✓ Experiment 3 complete. {len(df3)} records.')


# =============================================================================
# 7. Save CSVs
# =============================================================================

df1.to_csv('sk_exp1_single_run.csv', index=False)
df2.to_csv('sk_exp2_fixed_budget.csv', index=False)
df3.to_csv('sk_exp3_restart_sweep.csv', index=False)
print('Saved sk_exp1_single_run.csv, sk_exp2_fixed_budget.csv, sk_exp3_restart_sweep.csv')


# =============================================================================
# 8. Summary Tables
# =============================================================================

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean Relative Gap to Parisi Value (%)')
print(f'(single run per instance, averaged over {NUM_SEEDS} seeds)')
print('='*70)
gap1 = df1.groupby('iterations')[
    ['gd_gap_pct', 'sgd_gap_pct', 'amp_gap_pct', 'spec_gap_pct']
].mean().round(2)
gap1.columns = ['GD', 'SGD', 'AMP', 'Spectral']
print(gap1.to_string())

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean energy/N by N and iteration count')
print('='*70)
for iter_val in ITERATION_VALUES:
    sub = df1[df1['iterations'] == iter_val]
    print(f'\n  iter={iter_val}:')
    for _, row in sub.iterrows():
        print(f"    N={int(row.N):>5d}  "
              f"GD: {row.gd_mean_eN:+.4f}  "
              f"SGD: {row.sgd_mean_eN:+.4f}  "
              f"AMP: {row.amp_mean_eN:+.4f}  "
              f"SPEC: {row.spec_mean_eN:+.4f}  "
              f"PARISI: {PARISI_VALUE:+.4f}")

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean wall time (s) per single run')
print('='*70)
time1 = df1.groupby('iterations')[
    ['gd_wall_sec', 'sgd_wall_sec', 'amp_wall_sec', 'spec_wall_sec']
].mean().round(4)
time1.columns = ['GD', 'SGD', 'AMP', 'Spectral']
print(time1.to_string())

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean FLOPs per single run')
print('='*70)
flop1 = df1.groupby('iterations')[
    ['gd_flops', 'sgd_flops', 'amp_flops', 'spec_flops']
].mean()
flop1.columns = ['GD', 'SGD', 'AMP', 'Spectral']
flop1_fmt = flop1.copy()
for col in flop1_fmt.columns:
    flop1_fmt[col] = flop1_fmt[col].apply(lambda x: f'{x:.2e}')
print(flop1_fmt.to_string())

print('\n' + '='*70)
print(f'EXPERIMENT 2 SUMMARY — Best gap within {TIME_BUDGET_SEC}s budget (%)')
print('='*70)
gap2 = df2.groupby('iterations')[
    ['gd_gap_pct', 'sgd_gap_pct', 'amp_gap_pct', 'spec_gap_pct']
].mean().round(2)
gap2.columns = ['GD', 'SGD', 'AMP', 'Spectral']
print(gap2.to_string())

print('\n' + '='*70)
print(f'EXPERIMENT 2 SUMMARY — Mean restarts completed within {TIME_BUDGET_SEC}s')
print('='*70)
restarts2 = df2.groupby('iterations')[
    ['gd_restarts', 'sgd_restarts', 'amp_restarts', 'spec_restarts']
].mean().round(1)
restarts2.columns = ['GD', 'SGD', 'AMP', 'Spectral']
print(restarts2.to_string())

print('\n' + '='*70)
print('EXPERIMENT 3 SUMMARY — AMP vs Spectral gap (%) by N and restart count')
print(f'(AMP uses {EXP3_AMP_ITERS} iterations/restart, averaged over {EXP3_NUM_SEEDS} seeds)')
print('='*70)
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N]
    print(f'\n  N={N}:')
    print(f"    {'restarts':>8}  {'AMP gap%':>9}  {'SPEC gap%':>10}  "
          f"{'AMP adv (pp)':>13}  {'AMP time(s)':>11}  "
          f"{'SPEC time(s)':>12}  {'time ratio':>10}  {'FLOP ratio':>10}")
    for _, row in sub.iterrows():
        adv = -row.quality_gap_pp   # positive = AMP is better (smaller gap)
        print(f"    {int(row.num_restarts):>8}  "
              f"{row.amp_gap_pct:>9.2f}  "
              f"{row.spec_gap_pct:>10.2f}  "
              f"{adv:>+13.2f}  "
              f"{row.amp_wall_sec:>11.3f}  "
              f"{row.spec_wall_sec:>12.4f}  "
              f"{row.wall_ratio:>10.1f}x  "
              f"{row.flop_ratio:>10.1f}x")

print('\n' + '='*70)
print('EXPERIMENT 3 SUMMARY — FLOP cost at each restart count')
print('='*70)
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N]
    print(f'\n  N={N}:')
    print(f"    {'restarts':>8}  {'AMP FLOPs':>12}  {'SPEC FLOPs':>12}  {'ratio':>8}")
    for _, row in sub.iterrows():
        print(f"    {int(row.num_restarts):>8}  "
              f"{row.amp_flops:>12.3e}  "
              f"{row.spec_flops:>12.3e}  "
              f"{row.flop_ratio:>8.1f}x")

print('\n' + '='*70)
print('Win rates — Experiment 1 (single run)')
print('='*70)
print(df1.groupby('iterations')['winner'].value_counts(normalize=True)
      .mul(100).round(1).to_string())

print('\n' + '='*70)
print('Win rates — Experiment 2 (fixed budget)')
print('='*70)
print(df2.groupby('iterations')['winner'].value_counts(normalize=True)
      .mul(100).round(1).to_string())

print('\n' + '='*70)
print('Win rates — Experiment 3 (restart sweep, AMP vs Spectral only)')
print('='*70)
print(df3.groupby('num_restarts')['winner'].value_counts(normalize=True)
      .mul(100).round(1).to_string())


# =============================================================================
# 9. Visualizations
# =============================================================================

# ── Figure 1: Exp 1 — Gap vs N (with std shading) ────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle('Experiment 1: Single-Run Gap to Parisi Value (mean ± 1 std)',
             fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df1[df1['iterations'] == iter_val]
    for algo, col_mean, col_std, marker in [
        ('GD',      'gd_gap_pct',   'gd_std_eN',   'o'),
        ('SGD',     'sgd_gap_pct',  'sgd_std_eN',  '^'),
        ('AMP',     'amp_gap_pct',  'amp_std_eN',  's'),
        ('Spectral','spec_gap_pct', 'spec_std_eN', 'D'),
    ]:
        key = 'SPEC' if algo == 'Spectral' else algo
        means = sub[col_mean].values
        stds  = sub[col_std].values * 100
        ax.plot(sub['N'], means, marker=marker, lw=2, color=ALGO_COLORS[key], label=algo)
        ax.fill_between(sub['N'], means - stds, means + stds, alpha=0.15, color=ALGO_COLORS[key])
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Relative gap to Parisi (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
plt.tight_layout()
plt.savefig('fig_exp1_gap_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_gap_vs_N.png')


# ── Figure 2: Exp 2 — Gap vs N under fixed time budget ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle(f'Experiment 2: Best Gap within {TIME_BUDGET_SEC}s Budget',
             fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df2[df2['iterations'] == iter_val]
    ax.semilogy(sub['N'], sub['gd_gap_pct'],   'o-', color=ALGO_COLORS['GD'],   label='GD',       lw=2)
    ax.semilogy(sub['N'], sub['sgd_gap_pct'],  '^-', color=ALGO_COLORS['SGD'],  label='SGD',      lw=2)
    ax.semilogy(sub['N'], sub['amp_gap_pct'],  's-', color=ALGO_COLORS['AMP'],  label='AMP',      lw=2)
    ax.semilogy(sub['N'], sub['spec_gap_pct'], 'D-', color=ALGO_COLORS['SPEC'], label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Relative gap to Parisi (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp2_gap_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp2_gap_vs_N.png')


# ── Figure 3: Exp 2 — Restarts completed per algorithm vs N ──────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
fig.suptitle(f'Experiment 2: Restarts Completed within {TIME_BUDGET_SEC}s',
             fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df2[df2['iterations'] == iter_val]
    ax.semilogy(sub['N'], sub['gd_restarts'],   'o-', color=ALGO_COLORS['GD'],   label='GD',       lw=2)
    ax.semilogy(sub['N'], sub['sgd_restarts'],  '^-', color=ALGO_COLORS['SGD'],  label='SGD',      lw=2)
    ax.semilogy(sub['N'], sub['amp_restarts'],  's-', color=ALGO_COLORS['AMP'],  label='AMP',      lw=2)
    ax.semilogy(sub['N'], sub['spec_restarts'], 'D-', color=ALGO_COLORS['SPEC'], label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Restarts within budget')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp2_restarts_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp2_restarts_vs_N.png')


# ── Figure 4: Exp 1 vs Exp 2 side-by-side at middle N ────────────────────────
compare_N = N_VALUES[len(N_VALUES) // 2]
sub1 = df1[df1['N'] == compare_N]
sub2 = df2[df2['N'] == compare_N]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle(f'Exp 1 vs Exp 2: Gap at N={compare_N} (lower = better)',
             fontsize=13, fontweight='bold')
x = np.arange(len(ITERATION_VALUES))
width = 0.20
for ax, sub, title in [(ax1, sub1, 'Exp 1: Single run per seed'),
                        (ax2, sub2, f'Exp 2: Best within {TIME_BUDGET_SEC}s')]:
    for offset, algo, col, key in [
        (-1.5*width, 'GD',       'gd_gap_pct',   'GD'),
        (-0.5*width, 'SGD',      'sgd_gap_pct',  'SGD'),
        ( 0.5*width, 'AMP',      'amp_gap_pct',  'AMP'),
        ( 1.5*width, 'Spectral', 'spec_gap_pct', 'SPEC'),
    ]:
        vals = [float(sub[sub['iterations']==i][col].values[0]) for i in ITERATION_VALUES]
        ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS[key], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ITERATION_VALUES])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Relative gap to Parisi (%)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_exp1_vs_exp2_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_vs_exp2_bar.png')


# ── Figure 5: Exp 1 — Variance comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle('Experiment 1: Variance of Single-Run Energy/N Across Seeds (std)',
             fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df1[df1['iterations'] == iter_val]
    ax.plot(sub['N'], sub['gd_std_eN'],   'o-', color=ALGO_COLORS['GD'],   label='GD',       lw=2)
    ax.plot(sub['N'], sub['sgd_std_eN'],  '^-', color=ALGO_COLORS['SGD'],  label='SGD',      lw=2)
    ax.plot(sub['N'], sub['amp_std_eN'],  's-', color=ALGO_COLORS['AMP'],  label='AMP',      lw=2)
    ax.plot(sub['N'], sub['spec_std_eN'], 'D-', color=ALGO_COLORS['SPEC'], label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Std of energy/N across seeds')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_exp1_variance_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_variance_vs_N.png')


# ── Figure 6: Exp 3 — Gap vs restarts, one line per N ────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Experiment 3: AMP vs Spectral — Gap to Parisi vs Restart Budget\n'
             f'(AMP: {EXP3_AMP_ITERS} iters/restart, averaged over {EXP3_NUM_SEEDS} seeds)',
             fontsize=12, fontweight='bold')

for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    ax1.plot(sub['num_restarts'], sub['amp_gap_pct'],
             marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}', linestyle='-')
    ax2.plot(sub['num_restarts'], sub['spec_gap_pct'],
             marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}', linestyle='--')

ax1.set_title('AMP', fontsize=12)
ax2.set_title('Spectral', fontsize=12)
for ax in (ax1, ax2):
    ax.axhline(0, color='k', lw=0.8, alpha=0.3)
    ax.set_xlabel('Number of restarts')
    ax.set_ylabel('Relative gap to Parisi (%)')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_gap_vs_restarts.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_gap_vs_restarts.png')


# ── Figure 7: Exp 3 — AMP advantage (pp) vs restarts ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Experiment 3: AMP Quality Advantage over Spectral vs Restart Budget\n'
             '(positive = AMP achieves smaller gap; negative = Spectral wins)',
             fontsize=11, fontweight='bold')
ax.axhline(0, color='k', lw=1.2, ls='--', alpha=0.5, label='Spectral = AMP')
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    advantage = -sub['quality_gap_pp'].values   # positive = AMP closer to Parisi
    ax.plot(sub['num_restarts'], advantage,
            marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
ax.set_xlabel('Number of restarts')
ax.set_ylabel('AMP advantage (percentage points of gap)')
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_amp_advantage.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_amp_advantage.png')


# ── Figure 8: Exp 3 — Quality vs FLOPs Pareto (the money plot) ───────────────
fig, axes = plt.subplots(1, len(EXP3_N_VALUES), figsize=(18, 5), sharey=True)
fig.suptitle('Experiment 3: Quality vs Total FLOPs — AMP vs Spectral Pareto Frontier\n'
             '(lower-left = better; each point = one restart count)',
             fontsize=12, fontweight='bold')

for ax, N in zip(axes, EXP3_N_VALUES):
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    ax.plot(sub['amp_flops'],  sub['amp_gap_pct'],  's-',
            color=ALGO_COLORS['AMP'],  lw=2, label='AMP',      markersize=8)
    ax.plot(sub['spec_flops'], sub['spec_gap_pct'], 'D-',
            color=ALGO_COLORS['SPEC'], lw=2, label='Spectral',  markersize=8)
    # Annotate each point with its restart count
    for _, row in sub.iterrows():
        ax.annotate(f'r={int(row.num_restarts)}',
                    (row.amp_flops,  row.amp_gap_pct),
                    fontsize=7, color=ALGO_COLORS['AMP'],
                    xytext=(4, 3), textcoords='offset points')
        ax.annotate(f'r={int(row.num_restarts)}',
                    (row.spec_flops, row.spec_gap_pct),
                    fontsize=7, color=ALGO_COLORS['SPEC'],
                    xytext=(4, -8), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_title(f'N={N}', fontsize=11)
    ax.set_xlabel('Total FLOPs')
    ax.set_ylabel('Relative gap to Parisi (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_pareto.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_pareto.png')


# ── Figure 9: Exp 3 — Wall time vs restarts ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Experiment 3: Wall-Clock Time vs Restart Budget',
             fontsize=12, fontweight='bold')
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    ax1.plot(sub['num_restarts'], sub['amp_wall_sec'],
             marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
    ax2.plot(sub['num_restarts'], sub['spec_wall_sec'],
             marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
ax1.set_title('AMP wall time', fontsize=11)
ax2.set_title('Spectral wall time', fontsize=11)
for ax in (ax1, ax2):
    ax.set_xlabel('Number of restarts')
    ax.set_ylabel('Wall time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_wall_time.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_wall_time.png')


# ── Figure 10: Exp 3 — FLOP ratio AMP/Spectral vs restarts ──────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle('Experiment 3: FLOP Ratio (AMP / Spectral) vs Restart Budget\n'
             '(how many times more FLOPs AMP spends to match Spectral restart count)',
             fontsize=11, fontweight='bold')
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    ax.plot(sub['num_restarts'], sub['flop_ratio'],
            marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
ax.axhline(1, color='k', lw=1, ls='--', alpha=0.4)
ax.set_xlabel('Number of restarts')
ax.set_ylabel('FLOP ratio (AMP / Spectral)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_flop_ratio.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_flop_ratio.png')


print('\n✓ All figures saved.')
print('\nKey output files:')
print('  sk_exp1_single_run.csv       — Experiment 1 data')
print('  sk_exp2_fixed_budget.csv     — Experiment 2 data')
print('  sk_exp3_restart_sweep.csv    — Experiment 3 data')
print('  fig_exp1_gap_vs_N.png        — Exp 1: gap with uncertainty bands')
print('  fig_exp2_gap_vs_N.png        — Exp 2: gap under equal time budget')
print('  fig_exp2_restarts_vs_N.png   — Exp 2: restarts each algo completed')
print('  fig_exp1_vs_exp2_bar.png     — Side-by-side Exp 1 vs Exp 2')
print('  fig_exp1_variance_vs_N.png   — Exp 1: per-algorithm variance')
print('  fig_exp3_gap_vs_restarts.png — Exp 3: gap vs restart count')
print('  fig_exp3_amp_advantage.png   — Exp 3: AMP advantage in pp over Spectral')
print('  fig_exp3_pareto.png          — Exp 3: quality vs FLOPs Pareto (money plot)')
print('  fig_exp3_wall_time.png       — Exp 3: wall time vs restarts')
print('  fig_exp3_flop_ratio.png      — Exp 3: FLOP ratio AMP/Spectral vs restarts')