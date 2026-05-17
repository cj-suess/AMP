import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from scipy.sparse.linalg import eigsh, LinearOperator
import os

warnings.filterwarnings('ignore')

print('Libraries loaded.')


# Algorithms

def generate_sk_matrix(N, rng):
    # sample upper triangle once, mirror, and force J_ii = 0
    J = np.zeros((N, N), dtype=float)
    iu = np.triu_indices(N, k=1)
    vals = rng.normal(0.0, 1.0 / np.sqrt(N), size=iu[0].shape[0])
    J[iu] = vals
    J[(iu[1], iu[0])] = vals
    np.fill_diagonal(J, 0.0)
    return J

def calculate_energy(sigma, J):
    # Montanari-style SK sign convention --> H_N(sigma) = -0.5 * sigma^T J sigma, with sigma_i in {-1,+1}
    return -0.5 * (sigma @ J @ sigma)

def energy_per_spin(sigma, J):
    return calculate_energy(sigma, J) / len(sigma)

# GD

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


# AMP Variants

def amp_sk(J, m_init, num_iterations, damping, beta_max=2.0):
    """Damped schedule with Onsager term using beta_{t-1}."""
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    betas = np.linspace(0.1, beta_max, num_iterations)
    beta_prev = betas[0]
    onsager_prev = np.mean(1.0 - m**2)
    for beta in betas:
        onsager = np.mean(1.0 - m**2)
        h_target = J @ m - beta_prev * onsager_prev * m_old
        h = damping * h + (1.0 - damping) * h_target
        m_old = np.copy(m)
        m = np.tanh(beta * h)
        beta_prev = beta
        onsager_prev = onsager
    return np.sign(m)


def amp_traditional_sk(J, m_init, num_iterations, beta_max=2.0):
    """Traditional AMP without damping; uses beta_{t-1} Onsager coefficient."""
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    betas = np.linspace(0.1, beta_max, num_iterations)
    beta_prev = betas[0]
    onsager_prev = np.mean(1.0 - m**2)
    for beta in betas:
        h = J @ m - beta_prev * onsager_prev * m_old
        m_new = np.tanh(beta * h)
        onsager_prev = np.mean(1.0 - m_new**2)
        beta_prev = beta
        m_old, m = m, m_new
    return np.sign(m)


# AMP traced variants (return per-iteration energy for trajectory plots)

def amp_sk_traced(J, m_init, num_iterations, damping, beta_max=2.0):
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    betas = np.linspace(0.1, beta_max, num_iterations)
    beta_prev = betas[0]
    onsager_prev = np.mean(1.0 - m**2)
    traj = []
    for beta in betas:
        onsager = np.mean(1.0 - m**2)
        h_target = J @ m - beta_prev * onsager_prev * m_old
        h = damping * h + (1.0 - damping) * h_target
        m_old = np.copy(m)
        m = np.tanh(beta * h)
        beta_prev = beta
        onsager_prev = onsager
        traj.append(calculate_energy(m, J) / N)
    return np.sign(m), traj


def amp_traditional_sk_traced(J, m_init, num_iterations, beta_max=2.0):
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    betas = np.linspace(0.1, beta_max, num_iterations)
    beta_prev = betas[0]
    onsager_prev = np.mean(1.0 - m**2)
    traj = []
    for beta in betas:
        h = J @ m - beta_prev * onsager_prev * m_old
        m_new = np.tanh(beta * h)
        onsager_prev = np.mean(1.0 - m_new**2)
        beta_prev = beta
        m_old, m = m, m_new
        traj.append(calculate_energy(m, J) / N)
    return np.sign(m), traj


def iamp_sk_traced(J, m_init, num_iterations, damping, x_grid, dPhi_dx):
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    traj = []

    def policy(k, field):
        grad = np.interp(field, x_grid, dPhi_dx[k],
                         left=dPhi_dx[k, 0], right=dPhi_dx[k, -1])
        return np.tanh(grad)

    beta_prev = 0.1
    onsager_prev = np.mean(1.0 - m**2)
    for k in range(num_iterations):
        h_target = J @ m - beta_prev * onsager_prev * m_old
        h = damping * h + (1.0 - damping) * h_target
        m_new = policy(k, h)
        onsager_prev = np.mean(1.0 - m_new**2)
        beta_prev = k / max(num_iterations - 1, 1)
        m_old, m = m, m_new
        traj.append(calculate_energy(m, J) / N)
    return np.sign(m), traj


def solve_parisi_pde_sk(mu_t, x_grid, t_grid, cfl_safety=0.4):
    """Backward solve of the Parisi PDE for SK (xi''(t)=1) via Cole-Hopf.

    The PDE u_t = -0.5 * (u_xx + mu(t) u_x^2) is solved backward from
    u(T, x) = log(2 cosh(x)) down to t=0. For piecewise-constant mu, the
    Cole-Hopf substitution v = exp(mu u) linearizes it to a backward heat
    equation v_t = -0.5 v_xx, which is unconditionally stable in forward
    backward-time tau = T - t under any explicit scheme satisfying CFL.

    We treat mu as piecewise-constant on each [t_grid[k], t_grid[k+1]] using
    its right endpoint value, and re-transform u <-> v at each grid step.
    """
    nx = len(x_grid)
    nt = len(t_grid)
    dx = x_grid[1] - x_grid[0]
    dt_max = cfl_safety * dx * dx
    eps = 1e-12

    Phi = np.zeros((nt, nx), dtype=float)
    Phi[-1] = np.log(2.0 * np.cosh(x_grid))

    u = Phi[-1].copy()
    for k in range(nt - 2, -1, -1):
        dt_total = t_grid[k + 1] - t_grid[k]
        mu_seg = max(mu_t[k + 1], eps)
        v = np.exp(mu_seg * u)
        n_sub = max(1, int(np.ceil(dt_total / dt_max)))
        dt_sub = dt_total / n_sub
        for _ in range(n_sub):
            vxx = np.gradient(np.gradient(v, dx, edge_order=2), dx, edge_order=2)
            v = v + dt_sub * 0.5 * vxx
            v[0] = v[1]
            v[-1] = v[-2]
            v = np.maximum(v, eps)
        u = np.log(v) / mu_seg
        Phi[k] = u

    dPhi_dx = np.gradient(Phi, dx, axis=1, edge_order=2)
    return Phi, dPhi_dx


def iamp_sk(J, m_init, num_iterations, damping, x_grid, dPhi_dx):
    """IAMP with Parisi-PDE-driven nonlinearity (Montanari-style control policy).

    If x_grid and dPhi_dx are provided, we skip the PDE solve and use the cached
    arrays. Otherwise solves the PDE inline. The cached path lets a restart
    sweep reuse one PDE solve across many runs at the same num_iterations.
    """
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)

    def policy(k, field):
        grad = np.interp(field, x_grid, dPhi_dx[k], left=dPhi_dx[k, 0], right=dPhi_dx[k, -1])
        return np.tanh(grad)

    beta_prev = 0.1
    onsager_prev = np.mean(1.0 - m**2)
    for k in range(num_iterations):
        h_target = J @ m - beta_prev * onsager_prev * m_old
        h = damping * h + (1.0 - damping) * h_target
        m_new = policy(k, h)
        onsager_prev = np.mean(1.0 - m_new**2)
        beta_prev = k / max(num_iterations - 1, 1)
        m_old, m = m, m_new

    return np.sign(m)

def optimize_parisi_measure_sk(num_iterations, pde_xmax=6.0, pde_nx=401, opt_steps=30, lr=0.5):
    """Numerically minimizes the zero-temperature Parisi functional via finite differences."""
    t_grid = np.linspace(0.0, 1.0, num_iterations)
    x_grid = np.linspace(-pde_xmax, pde_xmax, pde_nx)
    mu_t = np.linspace(0.0, 1.0, num_iterations)

    def eval_parisi(mu_array):
        mu_valid = np.maximum.accumulate(np.clip(mu_array, 0.0, 1.0))
        Phi, _ = solve_parisi_pde_sk(mu_valid, x_grid, t_grid)
        phi_0_0 = np.interp(0.0, x_grid, Phi[0])
        integral = 0.5 * np.trapz(t_grid * mu_valid, t_grid)
        return phi_0_0 - integral

    eps = 1e-4
    for _ in range(opt_steps):
        grad = np.zeros(num_iterations)
        base_val = eval_parisi(mu_t)
        for i in range(1, num_iterations):
            mu_eps = np.copy(mu_t)
            mu_eps[i] += eps
            grad[i] = (eval_parisi(mu_eps) - base_val) / eps
        mu_t -= lr * grad
        mu_t = np.maximum.accumulate(np.clip(mu_t, 0.0, 1.0))

    return mu_t


def precompute_iamp_pde(num_iterations, pde_xmax=6.0, pde_nx=401):
    t_grid = np.linspace(0.0, 1.0, num_iterations)
    x_grid = np.linspace(-pde_xmax, pde_xmax, pde_nx)
    cache_filename = f"parisi_mu_t_{num_iterations}_{pde_nx}.npy"
    if os.path.exists(cache_filename):
        print(f"Loading cached Parisi measure from {cache_filename}...")
        mu_t = np.load(cache_filename)
    else:
        print("Optimizing Parisi measure (this may take a minute)...")
        mu_t = optimize_parisi_measure_sk(num_iterations, pde_xmax, pde_nx)
        np.save(cache_filename, mu_t)
    _, dPhi_dx = solve_parisi_pde_sk(mu_t, x_grid, t_grid)
    return x_grid, dPhi_dx

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


def project_to_spins_and_quench(sigma, J):
    spins = np.sign(np.copy(sigma).astype(float))
    spins[spins == 0] = 1.0
    return greedy_quench(spins, J)

def get_orthogonal_starts(num_starts, N, rng, scale=0.001):
    """Generate `num_starts` near-center starts that are orthogonal in blocks."""
    starts = np.zeros((num_starts, N), dtype=float)
    filled = 0
    while filled < num_starts:
        block = min(N, num_starts - filled)
        A = rng.normal(size=(N, block))
        Q, _ = np.linalg.qr(A)
        starts[filled:filled + block] = (Q.T * (scale / np.sqrt(N)))
        filled += block
    return starts

# Spectral

def spectral_sk(J, refine=True):
    N = J.shape[0]
    # Wrap J in a LinearOperator to count actual matrix-vector products used
    # by ARPACK, replacing the sqrt(N) heuristic with an exact count.
    matvec_count = [0]

    def mv(x):
        matvec_count[0] += 1
        return J @ x

    A = LinearOperator((N, N), matvec=mv, dtype=J.dtype)
    eigenvalues, eigenvectors = eigsh(
        A, k=1, which='LA', return_eigenvectors=True,
        maxiter=1000, tol=1e-6,
    )
    v = eigenvectors[:, 0]
    spins = np.sign(v)
    spins[spins == 0] = 1.0
    quench_passes = 0
    if refine:
        spins, quench_passes = greedy_quench(spins, J)
    info = {
        'top_eigenvalue':  float(eigenvalues[0]),
        'lanczos_matvecs': matvec_count[0],
        'quench_passes':   quench_passes,
        'eigenvector':     v,
    }
    return spins, info

print('Algorithm definitions ready.')

# FLOP Estimators

def flops_gd(N, iters, quench_passes=0):
    return iters * 2 * (2 * N * N) + quench_passes * (2 * N * N)

def flops_amp_single(N, amp_iters, quench_passes):
    return amp_iters * (2 * N * N) + quench_passes * (2 * N * N)

def flops_iamp_single(N, iamp_iters, quench_passes, pde_nx, pde_nt,
                      include_pde=True):
    """IAMP FLOPs: matvec + per-iter interp + PDE solve (if not cached)."""
    matvec_cost = iamp_iters * (2 * N * N)
    quench_cost = quench_passes * (2 * N * N)
    interp_cost = iamp_iters * N * max(1, int(np.log2(max(pde_nx, 2))))
    pde_cost = pde_nt * pde_nx * 5 if include_pde else 0
    return matvec_cost + quench_cost + interp_cost + pde_cost

def flops_spectral(N, lanczos_matvecs, quench_passes):
    # lanczos_matvecs is the actual count from the LinearOperator wrapper.
    return (lanczos_matvecs + quench_passes + 1) * (2 * N * N)

print('FLOP estimators defined.')


####################################################################################################
####################################################################################################

# Everything below is experiments and plotting (some plotting was experimental and not used)

####################################################################################################
####################################################################################################

"""
Experiment 1 — Single-Run Quality
    Each algorithm runs exactly once per GOE instance, averaged over
    NUM_SEEDS instances. No algorithm receives more attempts than another.
    Measures intrinsic per-run quality.

Experiment 2 — Fixed Wall-Clock Budget
    Each algorithm is given TIME_BUDGET_SEC seconds per (N, iterations)
    cell and runs as many restarts as fit within that budget, reporting
    the best result found. Measures practical value per unit of real time.

Experiment 3 — Restart Sweep (AMP variants vs Spectral)
    GD is excluded. AMP variants and Spectral are each given a fixed
    restart budget across a focused set of N values, with AMP using
    EXP3_AMP_ITERS iterations per restart. Measures how IAMP's quality
    advantage over Spectral grows with restart budget, and quantifies
    the FLOP cost of that advantage.
"""

# Configs

# Experiments 1 & 2
N_VALUES         = [100, 500, 1000, 5000]
ITERATION_VALUES = [100, 500, 1000]
NUM_SEEDS        = 10
TIME_BUDGET_SEC  = 2.0

# Experiment 3
EXP3_N_VALUES       = [500, 1000, 5000]
EXP3_RESTART_VALUES = [1, 10, 50, 100]
EXP3_AMP_ITERS      = 500
EXP3_NUM_SEEDS      = 5
PARISI_VALUE        = -0.7633

# Shared hyperparameters
GD_LR           = 0.1
GD_CONV_TOL     = 1e-5
AMP_DAMPING     = 0.7
AMP_INIT_SCALE  = 1e-3
IAMP_PDE_XMAX   = 6.0
IAMP_PDE_NX     = 401
SPECTRAL_REFINE = True

ALGO_COLORS = {
    'GD':    '#E91E63',
    'AMP':   '#00BCD4',
    'AMP-T': '#26A69A',
    'IAMP':  '#5C6BC0',
    'SPEC':  '#FF9800',
}
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
print(f'AMP init scale:          {AMP_INIT_SCALE}')
print(f'IAMP PDE grid:           nx={IAMP_PDE_NX}, xmax={IAMP_PDE_XMAX}')


# Experiment 1 — single-run quality, averaged over seeds

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

        gd_energies, amp_energies, ampt_energies, iamp_energies, spec_energies = (
            [], [], [], [], [])
        gd_walls, amp_walls, ampt_walls, iamp_walls, spec_walls = (
            [], [], [], [], [])
        gd_flops_list, amp_flops_list, ampt_flops_list, iamp_flops_list, spec_flops_list = (
            [], [], [], [], [])
        gd_conv_iters = []

        # IAMP PDE solve is shared across all seeds at this (N, ITER) cell
        iamp_x_grid, iamp_dPhi_dx = precompute_iamp_pde(
            ITER, pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX)

        for seed in range(NUM_SEEDS):
            rng = np.random.default_rng(seed)
            J = generate_sk_matrix(N, rng)

            # GD --> quench the continuous output before recording energy so it is on equal footing with AMP/Spectral
            t0 = time.perf_counter()
            sigma_gd, gd_curve, gd_conv_iter = gradient_descent_sk(
                J, ITER, GD_LR, GD_CONV_TOL, rng)
            gd_quenched, gd_qp = project_to_spins_and_quench(sigma_gd, J)
            gd_walls.append(time.perf_counter() - t0)
            gd_energies.append(float(calculate_energy(gd_quenched, J)))
            gd_flops_list.append(flops_gd(N, ITER, gd_qp))
            gd_conv_iters.append(gd_conv_iter)

            m_init = get_orthogonal_starts(1, N, rng, scale=AMP_INIT_SCALE)[0]
            t0 = time.perf_counter()
            raw = amp_sk(J, m_init, ITER, AMP_DAMPING)
            quenched, qp = greedy_quench(raw, J)
            amp_walls.append(time.perf_counter() - t0)
            amp_energies.append(float(calculate_energy(quenched, J)))
            amp_flops_list.append(flops_amp_single(N, ITER, qp))

            t0 = time.perf_counter()
            raw_t = amp_traditional_sk(J, m_init, ITER)
            quenched_t, qp_t = greedy_quench(raw_t, J)
            ampt_walls.append(time.perf_counter() - t0)
            ampt_energies.append(float(calculate_energy(quenched_t, J)))
            ampt_flops_list.append(flops_amp_single(N, ITER, qp_t))

            t0 = time.perf_counter()
            raw_i = iamp_sk(
                J, m_init, ITER, AMP_DAMPING,
                pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX,
                x_grid=iamp_x_grid, dPhi_dx=iamp_dPhi_dx)
            quenched_i, qp_i = greedy_quench(raw_i, J)
            iamp_walls.append(time.perf_counter() - t0)
            iamp_energies.append(float(calculate_energy(quenched_i, J)))
            iamp_flops_list.append(flops_iamp_single(
                N, ITER, qp_i, IAMP_PDE_NX, ITER,
                include_pde=(seed == 0)))

            t0 = time.perf_counter()
            spec_spins, spec_info = spectral_sk(J, refine=SPECTRAL_REFINE)
            spec_walls.append(time.perf_counter() - t0)
            spec_energies.append(float(calculate_energy(spec_spins, J)))
            spec_flops_list.append(flops_spectral(
                N, spec_info['lanczos_matvecs'], spec_info['quench_passes']))

        def stats(vals):
            return np.mean(vals), np.std(vals)

        gd_e_mean,   gd_e_std   = stats(gd_energies)
        amp_e_mean,  amp_e_std  = stats(amp_energies)
        ampt_e_mean, ampt_e_std = stats(ampt_energies)
        iamp_e_mean, iamp_e_std = stats(iamp_energies)
        spec_e_mean, spec_e_std = stats(spec_energies)

        def gap_pct(e): return 100 * abs(e - theoretical_limit) / abs(theoretical_limit)

        print(f'[{cell_idx:02d}/{total_cells}]  N={N:>5d}  iter={ITER:<5d}  '
              f'GD: {gd_e_mean/N:+.4f}  '
              f'AMP: {amp_e_mean/N:+.4f}  '
              f'AMP-T: {ampt_e_mean/N:+.4f}  '
              f'IAMP: {iamp_e_mean/N:+.4f}  '
              f'SPEC: {spec_e_mean/N:+.4f}  '
              f'PARISI: {PARISI_VALUE:+.4f}')

        exp1_records.append(dict(
            experiment        = 1,
            iterations        = ITER,
            N                 = N,
            theoretical_limit = round(theoretical_limit, 4),
            parisi_value      = PARISI_VALUE,
            gd_mean_eN        = round(gd_e_mean / N, 5),
            amp_mean_eN       = round(amp_e_mean / N, 5),
            ampt_mean_eN      = round(ampt_e_mean / N, 5),
            iamp_mean_eN      = round(iamp_e_mean / N, 5),
            spec_mean_eN      = round(spec_e_mean / N, 5),
            gd_std_eN         = round(gd_e_std / N, 5),
            amp_std_eN        = round(amp_e_std / N, 5),
            ampt_std_eN       = round(ampt_e_std / N, 5),
            iamp_std_eN       = round(iamp_e_std / N, 5),
            spec_std_eN       = round(spec_e_std / N, 5),
            gd_gap_pct        = round(gap_pct(gd_e_mean), 3),
            amp_gap_pct       = round(gap_pct(amp_e_mean), 3),
            ampt_gap_pct      = round(gap_pct(ampt_e_mean), 3),
            iamp_gap_pct      = round(gap_pct(iamp_e_mean), 3),
            spec_gap_pct      = round(gap_pct(spec_e_mean), 3),
            gd_wall_sec       = round(np.mean(gd_walls), 5),
            amp_wall_sec      = round(np.mean(amp_walls), 5),
            ampt_wall_sec     = round(np.mean(ampt_walls), 5),
            iamp_wall_sec     = round(np.mean(iamp_walls), 5),
            spec_wall_sec     = round(np.mean(spec_walls), 5),
            gd_flops          = int(np.mean(gd_flops_list)),
            amp_flops         = int(np.mean(amp_flops_list)),
            ampt_flops        = int(np.mean(ampt_flops_list)),
            iamp_flops        = int(np.mean(iamp_flops_list)),
            spec_flops        = int(np.mean(spec_flops_list)),
            winner            = min(
                [('GD', gd_e_mean),
                 ('AMP', amp_e_mean), ('AMP-T', ampt_e_mean),
                 ('IAMP', iamp_e_mean), ('SPEC', spec_e_mean)],
                key=lambda x: x[1])[0],
            num_seeds         = NUM_SEEDS,
        ))

df1 = pd.DataFrame(exp1_records)
print(f'\n✓ Experiment 1 complete. {len(df1)} records.')


# Experiment 2 — Fixed wall-clock budget

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

        # GD: quench each restart before comparing energies.
        gd_best = np.inf; gd_restarts = 0; gd_total_flops = 0
        t_gd_start = time.perf_counter()
        while time.perf_counter() - t_gd_start < TIME_BUDGET_SEC:
            sigma_gd, _, _ = gradient_descent_sk(J, ITER, GD_LR, GD_CONV_TOL, rng)
            gd_q, gd_qp = project_to_spins_and_quench(sigma_gd, J)
            e = float(calculate_energy(gd_q, J))
            if e < gd_best: gd_best = e
            gd_total_flops += flops_gd(N, ITER, gd_qp)
            gd_restarts += 1
        gd_wall = time.perf_counter() - t_gd_start

        amp_best = np.inf; amp_restarts = 0; amp_total_flops = 0
        amp_batch_size = min(N, 64)
        amp_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
        amp_start_idx = 0
        t_amp_start = time.perf_counter()
        while time.perf_counter() - t_amp_start < TIME_BUDGET_SEC:
            if amp_start_idx >= len(amp_starts):
                amp_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
                amp_start_idx = 0
            m_init = amp_starts[amp_start_idx]
            amp_start_idx += 1
            raw = amp_sk(J, m_init, ITER, AMP_DAMPING)
            quenched, qp = greedy_quench(raw, J)
            e = float(calculate_energy(quenched, J))
            if e < amp_best: amp_best = e
            amp_total_flops += flops_amp_single(N, ITER, qp)
            amp_restarts += 1
        amp_wall = time.perf_counter() - t_amp_start

        ampt_best = np.inf; ampt_restarts = 0; ampt_total_flops = 0
        ampt_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
        ampt_start_idx = 0
        t_ampt_start = time.perf_counter()
        while time.perf_counter() - t_ampt_start < TIME_BUDGET_SEC:
            if ampt_start_idx >= len(ampt_starts):
                ampt_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
                ampt_start_idx = 0
            m_init = ampt_starts[ampt_start_idx]
            ampt_start_idx += 1
            raw = amp_traditional_sk(J, m_init, ITER)
            quenched, qp = greedy_quench(raw, J)
            e = float(calculate_energy(quenched, J))
            if e < ampt_best: ampt_best = e
            ampt_total_flops += flops_amp_single(N, ITER, qp)
            ampt_restarts += 1
        ampt_wall = time.perf_counter() - t_ampt_start

        iamp_x_grid, iamp_dPhi_dx = precompute_iamp_pde(
            ITER, pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX)
        iamp_best = np.inf; iamp_restarts = 0; iamp_total_flops = 0
        iamp_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
        iamp_start_idx = 0
        t_iamp_start = time.perf_counter()
        while time.perf_counter() - t_iamp_start < TIME_BUDGET_SEC:
            if iamp_start_idx >= len(iamp_starts):
                iamp_starts = get_orthogonal_starts(amp_batch_size, N, rng, scale=AMP_INIT_SCALE)
                iamp_start_idx = 0
            m_init = iamp_starts[iamp_start_idx]
            iamp_start_idx += 1
            raw = iamp_sk(
                J, m_init, ITER, AMP_DAMPING,
                pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX,
                x_grid=iamp_x_grid, dPhi_dx=iamp_dPhi_dx)
            quenched, qp = greedy_quench(raw, J)
            e = float(calculate_energy(quenched, J))
            if e < iamp_best: iamp_best = e
            iamp_total_flops += flops_iamp_single(
                N, ITER, qp, IAMP_PDE_NX, ITER,
                include_pde=(iamp_restarts == 0))
            iamp_restarts += 1
        iamp_wall = time.perf_counter() - t_iamp_start

        # Spectral --> compute eigenvector once, then restart with sign flips
        spec_spins_init, spec_info_init = spectral_sk(J, refine=False)
        v = spec_info_init['eigenvector']
        spec_best = np.inf; spec_restarts = 0
        spec_total_flops = flops_spectral(N, spec_info_init['lanczos_matvecs'], 0)
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
              f'AMP: {amp_best/N:+.4f} (x{amp_restarts})  '
              f'AMP-T: {ampt_best/N:+.4f} (x{ampt_restarts})  '
              f'IAMP: {iamp_best/N:+.4f} (x{iamp_restarts})  '
              f'SPEC: {spec_best/N:+.4f} (x{spec_restarts})')

        exp2_records.append(dict(
            experiment        = 2,
            iterations        = ITER,
            N                 = N,
            time_budget_sec   = TIME_BUDGET_SEC,
            theoretical_limit = round(theoretical_limit, 4),
            parisi_value      = PARISI_VALUE,
            gd_best_eN        = round(gd_best / N, 5),
            amp_best_eN       = round(amp_best / N, 5),
            ampt_best_eN      = round(ampt_best / N, 5),
            iamp_best_eN      = round(iamp_best / N, 5),
            spec_best_eN      = round(spec_best / N, 5),
            gd_gap_pct        = round(gap_pct(gd_best), 3),
            amp_gap_pct       = round(gap_pct(amp_best), 3),
            ampt_gap_pct      = round(gap_pct(ampt_best), 3),
            iamp_gap_pct      = round(gap_pct(iamp_best), 3),
            spec_gap_pct      = round(gap_pct(spec_best), 3),
            gd_restarts       = gd_restarts,
            amp_restarts      = amp_restarts,
            ampt_restarts     = ampt_restarts,
            iamp_restarts     = iamp_restarts,
            spec_restarts     = spec_restarts,
            gd_wall_sec       = round(gd_wall, 3),
            amp_wall_sec      = round(amp_wall, 3),
            ampt_wall_sec     = round(ampt_wall, 3),
            iamp_wall_sec     = round(iamp_wall, 3),
            spec_wall_sec     = round(spec_wall, 3),
            gd_flops          = gd_total_flops,
            amp_flops         = amp_total_flops,
            ampt_flops        = ampt_total_flops,
            iamp_flops        = iamp_total_flops,
            spec_flops        = spec_total_flops,
            winner            = min(
                [('GD', gd_best),
                 ('AMP', amp_best), ('AMP-T', ampt_best),
                 ('IAMP', iamp_best), ('SPEC', spec_best)],
                key=lambda x: x[1])[0],
        ))

df2 = pd.DataFrame(exp2_records)
print(f'\n✓ Experiment 2 complete. {len(df2)} records.')


# Experiment 3 — AMP variants vs Spectral restart sweep

print('\n' + '='*70)
print('EXPERIMENT 3: AMP variants vs Spectral — Restart Budget Sweep')
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

        amp_best_list   = []
        ampt_best_list  = []
        iamp_best_list  = []
        spec_best_list  = []
        amp_wall_list   = []
        ampt_wall_list  = []
        iamp_wall_list  = []
        spec_wall_list  = []
        amp_flops_list  = []
        ampt_flops_list = []
        iamp_flops_list = []
        spec_flops_list = []

        iamp_x_grid, iamp_dPhi_dx = precompute_iamp_pde(
            EXP3_AMP_ITERS, pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX)

        for seed in range(EXP3_NUM_SEEDS):
            rng = np.random.default_rng(seed * 1000 + N)
            J = generate_sk_matrix(N, rng)

            amp_best = np.inf
            amp_total_flops = 0
            t0 = time.perf_counter()
            amp_starts = get_orthogonal_starts(num_restarts, N, rng, scale=AMP_INIT_SCALE)
            for m_init in amp_starts:
                raw = amp_sk(J, m_init, EXP3_AMP_ITERS, AMP_DAMPING)
                quenched, qp = greedy_quench(raw, J)
                e = float(calculate_energy(quenched, J))
                if e < amp_best:
                    amp_best = e
                amp_total_flops += flops_amp_single(N, EXP3_AMP_ITERS, qp)
            amp_wall = time.perf_counter() - t0

            ampt_best = np.inf
            ampt_total_flops = 0
            t0 = time.perf_counter()
            for m_init in amp_starts:
                raw = amp_traditional_sk(J, m_init, EXP3_AMP_ITERS)
                quenched, qp = greedy_quench(raw, J)
                e = float(calculate_energy(quenched, J))
                if e < ampt_best:
                    ampt_best = e
                ampt_total_flops += flops_amp_single(N, EXP3_AMP_ITERS, qp)
            ampt_wall = time.perf_counter() - t0

            iamp_best = np.inf
            iamp_total_flops = 0
            t0 = time.perf_counter()
            for r_idx, m_init in enumerate(amp_starts):
                raw = iamp_sk(
                    J, m_init, EXP3_AMP_ITERS, AMP_DAMPING,
                    pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX,
                    x_grid=iamp_x_grid, dPhi_dx=iamp_dPhi_dx)
                quenched, qp = greedy_quench(raw, J)
                e = float(calculate_energy(quenched, J))
                if e < iamp_best:
                    iamp_best = e
                iamp_total_flops += flops_iamp_single(
                    N, EXP3_AMP_ITERS, qp, IAMP_PDE_NX, EXP3_AMP_ITERS,
                    include_pde=(r_idx == 0))
            iamp_wall = time.perf_counter() - t0

            # Spectral --> compute eigenvector once per seed, then restart with sign flips
            t_eig = time.perf_counter()
            spec_spins_seed, spec_info_seed = spectral_sk(J, refine=False)
            eig_time = time.perf_counter() - t_eig
            v = spec_info_seed['eigenvector']

            spec_best = np.inf
            spec_total_flops = flops_spectral(N, spec_info_seed['lanczos_matvecs'], 0)
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
            ampt_best_list.append(ampt_best)
            iamp_best_list.append(iamp_best)
            spec_best_list.append(spec_best)
            amp_wall_list.append(amp_wall)
            ampt_wall_list.append(ampt_wall)
            iamp_wall_list.append(iamp_wall)
            spec_wall_list.append(spec_wall)
            amp_flops_list.append(amp_total_flops)
            ampt_flops_list.append(ampt_total_flops)
            iamp_flops_list.append(iamp_total_flops)
            spec_flops_list.append(spec_total_flops)

        amp_mean_eN  = np.mean(amp_best_list)  / N
        ampt_mean_eN = np.mean(ampt_best_list) / N
        iamp_mean_eN = np.mean(iamp_best_list) / N
        spec_mean_eN = np.mean(spec_best_list) / N
        amp_gap   = 100 * abs(np.mean(amp_best_list)  - theoretical_limit) / abs(theoretical_limit)
        ampt_gap  = 100 * abs(np.mean(ampt_best_list) - theoretical_limit) / abs(theoretical_limit)
        iamp_gap  = 100 * abs(np.mean(iamp_best_list) - theoretical_limit) / abs(theoretical_limit)
        spec_gap  = 100 * abs(np.mean(spec_best_list) - theoretical_limit) / abs(theoretical_limit)
        amp_wall_mean   = np.mean(amp_wall_list)
        ampt_wall_mean  = np.mean(ampt_wall_list)
        iamp_wall_mean  = np.mean(iamp_wall_list)
        spec_wall_mean  = np.mean(spec_wall_list)
        amp_flops_mean  = int(np.mean(amp_flops_list))
        ampt_flops_mean = int(np.mean(ampt_flops_list))
        iamp_flops_mean = int(np.mean(iamp_flops_list))
        spec_flops_mean = int(np.mean(spec_flops_list))
        flop_ratio      = amp_flops_mean  / max(spec_flops_mean, 1)
        ampt_flop_ratio = ampt_flops_mean / max(spec_flops_mean, 1)
        iamp_flop_ratio = iamp_flops_mean / max(spec_flops_mean, 1)
        quality_gap_pp      = amp_gap  - spec_gap
        ampt_quality_gap_pp = ampt_gap - spec_gap
        iamp_quality_gap_pp = iamp_gap - spec_gap

        winner = min(
            [('AMP', amp_mean_eN), ('AMP-T', ampt_mean_eN),
             ('IAMP', iamp_mean_eN), ('SPEC', spec_mean_eN)],
            key=lambda x: x[1])[0]

        print(f'[{cell_idx:02d}/{total_cells_3}]  N={N:>5d}  restarts={num_restarts:<4d}  '
              f'AMP: {amp_mean_eN:+.4f}  '
              f'AMP-T: {ampt_mean_eN:+.4f}  '
              f'IAMP: {iamp_mean_eN:+.4f}  '
              f'SPEC: {spec_mean_eN:+.4f}  '
              f'PARISI: {PARISI_VALUE:+.4f}  '
              f'| gap AMP/AMP-T/IAMP/SPEC: {amp_gap:.2f}/{ampt_gap:.2f}/{iamp_gap:.2f}/{spec_gap:.2f}%  '
              f'| t AMP/AMP-T/IAMP/SPEC: {amp_wall_mean:.2f}/{ampt_wall_mean:.2f}/{iamp_wall_mean:.2f}/{spec_wall_mean:.3f}s  '
              f'| FLOP ratios vs SPEC: {flop_ratio:.1f}/{ampt_flop_ratio:.1f}/{iamp_flop_ratio:.1f}x')

        exp3_records.append(dict(
            experiment          = 3,
            N                   = N,
            num_restarts        = num_restarts,
            amp_iters_per_run   = EXP3_AMP_ITERS,
            theoretical_limit   = round(theoretical_limit, 4),
            parisi_value        = PARISI_VALUE,
            amp_mean_eN         = round(amp_mean_eN, 5),
            ampt_mean_eN        = round(ampt_mean_eN, 5),
            iamp_mean_eN        = round(iamp_mean_eN, 5),
            spec_mean_eN        = round(spec_mean_eN, 5),
            amp_gap_pct         = round(amp_gap, 3),
            ampt_gap_pct        = round(ampt_gap, 3),
            iamp_gap_pct        = round(iamp_gap, 3),
            spec_gap_pct        = round(spec_gap, 3),
            quality_gap_pp      = round(quality_gap_pp, 3),
            ampt_quality_gap_pp = round(ampt_quality_gap_pp, 3),
            iamp_quality_gap_pp = round(iamp_quality_gap_pp, 3),
            amp_wall_sec        = round(amp_wall_mean, 4),
            ampt_wall_sec       = round(ampt_wall_mean, 4),
            iamp_wall_sec       = round(iamp_wall_mean, 4),
            spec_wall_sec       = round(spec_wall_mean, 4),
            wall_ratio          = round(amp_wall_mean  / max(spec_wall_mean, 1e-9), 2),
            ampt_wall_ratio     = round(ampt_wall_mean / max(spec_wall_mean, 1e-9), 2),
            iamp_wall_ratio     = round(iamp_wall_mean / max(spec_wall_mean, 1e-9), 2),
            amp_flops           = amp_flops_mean,
            ampt_flops          = ampt_flops_mean,
            iamp_flops          = iamp_flops_mean,
            spec_flops          = spec_flops_mean,
            flop_ratio          = round(flop_ratio, 1),
            ampt_flop_ratio     = round(ampt_flop_ratio, 1),
            iamp_flop_ratio     = round(iamp_flop_ratio, 1),
            winner              = winner,
            num_seeds           = EXP3_NUM_SEEDS,
        ))

df3 = pd.DataFrame(exp3_records)
print(f'\n✓ Experiment 3 complete. {len(df3)} records.')


# save CSVs

df1.to_csv('sk_exp1_single_run.csv', index=False)
df2.to_csv('sk_exp2_fixed_budget.csv', index=False)
df3.to_csv('sk_exp3_restart_sweep.csv', index=False)
print('Saved sk_exp1_single_run.csv, sk_exp2_fixed_budget.csv, sk_exp3_restart_sweep.csv')


# summary tables

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean Relative Gap to Parisi Value (%)')
print(f'(single run per instance, averaged over {NUM_SEEDS} seeds)')
print('='*70)
gap1 = df1.groupby('iterations')[
    ['gd_gap_pct', 'amp_gap_pct', 'ampt_gap_pct', 'iamp_gap_pct', 'spec_gap_pct']
].mean().round(2)
gap1.columns = ['GD', 'AMP', 'AMP-T', 'IAMP', 'Spectral']
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
              f"AMP: {row.amp_mean_eN:+.4f}  "
              f"AMP-T: {row.ampt_mean_eN:+.4f}  "
              f"IAMP: {row.iamp_mean_eN:+.4f}  "
              f"SPEC: {row.spec_mean_eN:+.4f}  "
              f"PARISI: {PARISI_VALUE:+.4f}")

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean wall time (s) per single run')
print('='*70)
time1 = df1.groupby('iterations')[
    ['gd_wall_sec', 'amp_wall_sec', 'ampt_wall_sec', 'iamp_wall_sec', 'spec_wall_sec']
].mean().round(4)
time1.columns = ['GD', 'AMP', 'AMP-T', 'IAMP', 'Spectral']
print(time1.to_string())

print('\n' + '='*70)
print('EXPERIMENT 1 SUMMARY — Mean FLOPs per single run')
print('='*70)
flop1 = df1.groupby('iterations')[
    ['gd_flops', 'amp_flops', 'ampt_flops', 'iamp_flops', 'spec_flops']
].mean()
flop1.columns = ['GD', 'AMP', 'AMP-T', 'IAMP', 'Spectral']
flop1_fmt = flop1.copy()
for col in flop1_fmt.columns:
    flop1_fmt[col] = flop1_fmt[col].apply(lambda x: f'{x:.2e}')
print(flop1_fmt.to_string())

print('\n' + '='*70)
print(f'EXPERIMENT 2 SUMMARY — Best gap within {TIME_BUDGET_SEC}s budget (%)')
print('='*70)
gap2 = df2.groupby('iterations')[
    ['gd_gap_pct', 'amp_gap_pct', 'ampt_gap_pct', 'iamp_gap_pct', 'spec_gap_pct']
].mean().round(2)
gap2.columns = ['GD', 'AMP', 'AMP-T', 'IAMP', 'Spectral']
print(gap2.to_string())

print('\n' + '='*70)
print(f'EXPERIMENT 2 SUMMARY — Mean restarts completed within {TIME_BUDGET_SEC}s')
print('='*70)
restarts2 = df2.groupby('iterations')[
    ['gd_restarts', 'amp_restarts', 'ampt_restarts', 'iamp_restarts', 'spec_restarts']
].mean().round(1)
restarts2.columns = ['GD', 'AMP', 'AMP-T', 'IAMP', 'Spectral']
print(restarts2.to_string())

print('\n' + '='*70)
print('EXPERIMENT 3 SUMMARY — AMP variants vs Spectral gap (%) by N and restart count')
print(f'(AMP/AMP-T/IAMP use {EXP3_AMP_ITERS} iterations/restart, averaged over {EXP3_NUM_SEEDS} seeds)')
print('='*70)
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N]
    print(f'\n  N={N}:')
    print(f"    {'restarts':>8}  {'AMP':>7}  {'AMP-T':>7}  {'IAMP':>7}  {'SPEC':>7}  "
          f"{'AMP adv':>9}  {'IAMP adv':>9}  {'AMP t(s)':>9}  {'IAMP t(s)':>10}  {'SPEC t(s)':>10}")
    for _, row in sub.iterrows():
        amp_adv  = -row.quality_gap_pp
        iamp_adv = -row.iamp_quality_gap_pp
        print(f"    {int(row.num_restarts):>8}  "
              f"{row.amp_gap_pct:>7.2f}  "
              f"{row.ampt_gap_pct:>7.2f}  "
              f"{row.iamp_gap_pct:>7.2f}  "
              f"{row.spec_gap_pct:>7.2f}  "
              f"{amp_adv:>+9.2f}  "
              f"{iamp_adv:>+9.2f}  "
              f"{row.amp_wall_sec:>9.3f}  "
              f"{row.iamp_wall_sec:>10.3f}  "
              f"{row.spec_wall_sec:>10.4f}")

print('\n' + '='*70)
print('EXPERIMENT 3 SUMMARY — FLOP cost at each restart count')
print('='*70)
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N]
    print(f'\n  N={N}:')
    print(f"    {'restarts':>8}  {'AMP FLOPs':>12}  {'AMP-T FLOPs':>12}  "
          f"{'IAMP FLOPs':>12}  {'SPEC FLOPs':>12}  {'AMP/SPEC':>9}  {'IAMP/SPEC':>10}")
    for _, row in sub.iterrows():
        print(f"    {int(row.num_restarts):>8}  "
              f"{row.amp_flops:>12.3e}  "
              f"{row.ampt_flops:>12.3e}  "
              f"{row.iamp_flops:>12.3e}  "
              f"{row.spec_flops:>12.3e}  "
              f"{row.flop_ratio:>9.1f}x  "
              f"{row.iamp_flop_ratio:>10.1f}x")

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


# trajectory data collection

TRAJ_N    = 1000
TRAJ_ITER = 500
TRAJ_SEED = 0

print(f'\nCollecting trajectory data  (N={TRAJ_N}, {TRAJ_ITER} iterations)...')
rng_traj    = np.random.default_rng(TRAJ_SEED)
J_traj      = generate_sk_matrix(TRAJ_N, rng_traj)
m_init_traj = get_orthogonal_starts(1, TRAJ_N, rng_traj, scale=AMP_INIT_SCALE)[0]

sigma_gd_traj, gd_traj_raw, _ = gradient_descent_sk(
    J_traj, TRAJ_ITER, GD_LR, GD_CONV_TOL, rng_traj)
gd_traj_eN = [e / TRAJ_N for e in gd_traj_raw]

iamp_x_traj, iamp_dphi_traj = precompute_iamp_pde(
    TRAJ_ITER, pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX)

_, amp_traj  = amp_sk_traced(J_traj, m_init_traj, TRAJ_ITER, AMP_DAMPING)
_, ampt_traj = amp_traditional_sk_traced(J_traj, m_init_traj, TRAJ_ITER)
_, iamp_traj = iamp_sk_traced(
    J_traj, m_init_traj, TRAJ_ITER, AMP_DAMPING, iamp_x_traj, iamp_dphi_traj)

spec_spins_traj, _ = spectral_sk(J_traj, refine=True)
spec_eN_traj = energy_per_spin(spec_spins_traj, J_traj)

print('Trajectory data collected.')


# cdf data collection

CDF_N     = 1000
CDF_ITER  = 500
CDF_SEEDS = 40

print(f'\nCollecting CDF data  (N={CDF_N}, {CDF_ITER} iterations, {CDF_SEEDS} seeds)...')

cdf_x_grid, cdf_dPhi_dx = precompute_iamp_pde(
    CDF_ITER, pde_xmax=IAMP_PDE_XMAX, pde_nx=IAMP_PDE_NX)

cdf_energies = {k: [] for k in ['GD', 'AMP', 'AMP-T', 'IAMP', 'SPEC']}

for seed in range(CDF_SEEDS):
    rng    = np.random.default_rng(seed + 9999)
    J      = generate_sk_matrix(CDF_N, rng)
    m_init = get_orthogonal_starts(1, CDF_N, rng, scale=AMP_INIT_SCALE)[0]

    sigma_gd, _, _ = gradient_descent_sk(J, CDF_ITER, GD_LR, GD_CONV_TOL, rng)
    gd_q, _ = project_to_spins_and_quench(sigma_gd, J)
    cdf_energies['GD'].append(energy_per_spin(gd_q, J))

    raw = amp_sk(J, m_init, CDF_ITER, AMP_DAMPING)
    q, _ = greedy_quench(raw, J)
    cdf_energies['AMP'].append(energy_per_spin(q, J))

    raw_t = amp_traditional_sk(J, m_init, CDF_ITER)
    q_t, _ = greedy_quench(raw_t, J)
    cdf_energies['AMP-T'].append(energy_per_spin(q_t, J))

    raw_i = iamp_sk(J, m_init, CDF_ITER, AMP_DAMPING,
                    x_grid=cdf_x_grid, dPhi_dx=cdf_dPhi_dx)
    q_i, _ = greedy_quench(raw_i, J)
    cdf_energies['IAMP'].append(energy_per_spin(q_i, J))

    sp, _ = spectral_sk(J, refine=True)
    cdf_energies['SPEC'].append(energy_per_spin(sp, J))

    if (seed + 1) % 10 == 0:
        print(f'  {seed + 1}/{CDF_SEEDS} seeds done')

print('CDF data collected.')


# plots

# Figure 1: Exp 1 — Gap vs N (with std shading)
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle('Exp 1: Gap to Parisi  (mean ± 1 std, single run)', fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df1[df1['iterations'] == iter_val]
    for algo, col_mean, col_std, marker in [
        ('GD',       'gd_gap_pct',   'gd_std_eN',   'o'),
        ('AMP',      'amp_gap_pct',  'amp_std_eN',  's'),
        ('AMP-T',    'ampt_gap_pct', 'ampt_std_eN', 'P'),
        ('IAMP',     'iamp_gap_pct', 'iamp_std_eN', 'X'),
        ('Spectral', 'spec_gap_pct', 'spec_std_eN', 'D'),
    ]:
        key = 'SPEC' if algo == 'Spectral' else algo
        means = sub[col_mean].values
        stds  = sub[col_std].values * 100
        ax.plot(sub['N'], means, marker=marker, lw=2, color=ALGO_COLORS[key], label=algo)
        ax.fill_between(sub['N'], means - stds, means + stds, alpha=0.20, color=ALGO_COLORS[key])
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Gap to Parisi (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
plt.tight_layout()
plt.savefig('fig_exp1_gap_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_gap_vs_N.png')


# Figure 2: Exp 2 — Gap vs N under fixed time budget
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle(f'Exp 2: Best gap within {TIME_BUDGET_SEC}s', fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df2[df2['iterations'] == iter_val]
    ax.semilogy(sub['N'], sub['gd_gap_pct'],   'o-', color=ALGO_COLORS['GD'],    label='GD',       lw=2)
    ax.semilogy(sub['N'], sub['amp_gap_pct'],  's-', color=ALGO_COLORS['AMP'],   label='AMP',      lw=2)
    ax.semilogy(sub['N'], sub['ampt_gap_pct'], 'P-', color=ALGO_COLORS['AMP-T'], label='AMP-T',    lw=2)
    ax.semilogy(sub['N'], sub['iamp_gap_pct'], 'X-', color=ALGO_COLORS['IAMP'],  label='IAMP',     lw=2)
    ax.semilogy(sub['N'], sub['spec_gap_pct'], 'D-', color=ALGO_COLORS['SPEC'],  label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Gap to Parisi (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp2_gap_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp2_gap_vs_N.png')


# Figure 3: Exp 2 — Restarts completed per algorithm vs N
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
fig.suptitle(f'Exp 2: Restarts completed within {TIME_BUDGET_SEC}s', fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df2[df2['iterations'] == iter_val]
    ax.semilogy(sub['N'], sub['gd_restarts'],   'o-', color=ALGO_COLORS['GD'],    label='GD',       lw=2)
    ax.semilogy(sub['N'], sub['amp_restarts'],  's-', color=ALGO_COLORS['AMP'],   label='AMP',      lw=2)
    ax.semilogy(sub['N'], sub['ampt_restarts'], 'P-', color=ALGO_COLORS['AMP-T'], label='AMP-T',    lw=2)
    ax.semilogy(sub['N'], sub['iamp_restarts'], 'X-', color=ALGO_COLORS['IAMP'],  label='IAMP',     lw=2)
    ax.semilogy(sub['N'], sub['spec_restarts'], 'D-', color=ALGO_COLORS['SPEC'],  label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Restarts within budget')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp2_restarts_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp2_restarts_vs_N.png')


# Figure 4: Exp 1 vs Exp 2 side-by-side bar
compare_N = N_VALUES[len(N_VALUES) // 2]
sub1 = df1[df1['N'] == compare_N]
sub2 = df2[df2['N'] == compare_N]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle(f'Exp 1 vs Exp 2: gap at N={compare_N}', fontsize=13, fontweight='bold')
x = np.arange(len(ITERATION_VALUES))
width = 0.13
for ax, sub, title in [(ax1, sub1, 'Exp 1: single run'),
                        (ax2, sub2, f'Exp 2: best within {TIME_BUDGET_SEC}s')]:
    for offset, algo, col, key in [
        (-2.0*width, 'GD',       'gd_gap_pct',   'GD'),
        (-1.0*width, 'AMP',      'amp_gap_pct',  'AMP'),
        ( 0.0*width, 'AMP-T',    'ampt_gap_pct', 'AMP-T'),
        ( 1.0*width, 'IAMP',     'iamp_gap_pct', 'IAMP'),
        ( 2.0*width, 'Spectral', 'spec_gap_pct', 'SPEC'),
    ]:
        vals = [float(sub[sub['iterations']==i][col].values[0]) for i in ITERATION_VALUES]
        ax.bar(x + offset, vals, width, label=algo, color=ALGO_COLORS[key], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ITERATION_VALUES])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gap to Parisi (%)')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_exp1_vs_exp2_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_vs_exp2_bar.png')


# Figure 5: Exp 1 — Variance comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle('Exp 1: Std of energy/N across seeds', fontsize=13, fontweight='bold')
for ax, iter_val in zip(axes, ITERATION_VALUES):
    sub = df1[df1['iterations'] == iter_val]
    ax.plot(sub['N'], sub['gd_std_eN'],   'o-', color=ALGO_COLORS['GD'],    label='GD',       lw=2)
    ax.plot(sub['N'], sub['amp_std_eN'],  's-', color=ALGO_COLORS['AMP'],   label='AMP',      lw=2)
    ax.plot(sub['N'], sub['ampt_std_eN'], 'P-', color=ALGO_COLORS['AMP-T'], label='AMP-T',    lw=2)
    ax.plot(sub['N'], sub['iamp_std_eN'], 'X-', color=ALGO_COLORS['IAMP'],  label='IAMP',     lw=2)
    ax.plot(sub['N'], sub['spec_std_eN'], 'D-', color=ALGO_COLORS['SPEC'],  label='Spectral', lw=2)
    ax.set_title(f'{iter_val} iterations', fontsize=11)
    ax.set_xlabel('N')
    ax.set_ylabel('Std of energy/N')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_exp1_variance_vs_N.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp1_variance_vs_N.png')


# Figure 6: Exp 3 — Gap vs restarts, one panel per AMP variant
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
fig.suptitle(f'Exp 3: Gap to Parisi vs restart budget  ({EXP3_AMP_ITERS} iters/restart)',
             fontsize=12, fontweight='bold')
panel_specs = [
    (axes[0], 'amp_gap_pct',  'AMP',      'AMP'),
    (axes[1], 'ampt_gap_pct', 'AMP-T',    'AMP-T'),
    (axes[2], 'iamp_gap_pct', 'IAMP',     'IAMP'),
    (axes[3], 'spec_gap_pct', 'Spectral', 'SPEC'),
]
for ax, col, title, key in panel_specs:
    for N in EXP3_N_VALUES:
        sub = df3[df3['N'] == N].sort_values('num_restarts')
        ax.plot(sub['num_restarts'], sub[col],
                marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
    ax.axhline(0, color='k', lw=0.8, alpha=0.3)
    ax.set_title(title, fontsize=12, color=ALGO_COLORS[key])
    ax.set_xlabel('Restarts')
    ax.set_ylabel('Gap to Parisi (%)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_gap_vs_restarts.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_gap_vs_restarts.png')


# Figure 7: Exp 3 — AMP/IAMP advantage over Spectral
fig, (ax_amp, ax_iamp) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
fig.suptitle('Exp 3: AMP and IAMP advantage over Spectral  (positive = closer to Parisi)',
             fontsize=11, fontweight='bold')
for ax, col, title, key in [
    (ax_amp,  'quality_gap_pp',      'AMP',  'AMP'),
    (ax_iamp, 'iamp_quality_gap_pp', 'IAMP', 'IAMP'),
]:
    ax.axhline(0, color='k', lw=1.2, ls='--', alpha=0.5, label='same as Spectral')
    for N in EXP3_N_VALUES:
        sub = df3[df3['N'] == N].sort_values('num_restarts')
        advantage = -sub[col].values
        ax.plot(sub['num_restarts'], advantage,
                marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
    ax.set_title(title, fontsize=11, color=ALGO_COLORS[key])
    ax.set_xlabel('Restarts')
    ax.set_ylabel('Advantage (pp of gap)')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_amp_advantage.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_amp_advantage.png')


# Figure 8: Exp 3 — Quality vs FLOPs
fig, axes = plt.subplots(1, len(EXP3_N_VALUES), figsize=(20, 5.5), sharey=True)
fig.suptitle('Exp 3: Quality vs FLOPs  (lower-left = better)',
             fontsize=12, fontweight='bold')
variant_specs = [
    ('amp_flops',  'amp_gap_pct',  'AMP',      'AMP',   's'),
    ('ampt_flops', 'ampt_gap_pct', 'AMP-T',    'AMP-T', 'P'),
    ('iamp_flops', 'iamp_gap_pct', 'IAMP',     'IAMP',  'X'),
    ('spec_flops', 'spec_gap_pct', 'Spectral', 'SPEC',  'D'),
]
for ax, N in zip(axes, EXP3_N_VALUES):
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    for flop_col, gap_col, label, key, marker in variant_specs:
        ax.plot(sub[flop_col], sub[gap_col], marker=marker, linestyle='-',
                color=ALGO_COLORS[key], lw=2, label=label, markersize=8)
    ax.set_xscale('log')
    ax.set_title(f'N={N}', fontsize=11)
    ax.set_xlabel('FLOPs')
    ax.set_ylabel('Gap to Parisi (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_pareto.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_pareto.png')


# Figure 9: Exp 3 — Wall time vs restarts
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig.suptitle('Exp 3: Wall time vs restart budget', fontsize=12, fontweight='bold')
panel_specs = [
    (axes[0], 'amp_wall_sec',  'AMP',      'AMP'),
    (axes[1], 'ampt_wall_sec', 'AMP-T',    'AMP-T'),
    (axes[2], 'iamp_wall_sec', 'IAMP',     'IAMP'),
    (axes[3], 'spec_wall_sec', 'Spectral', 'SPEC'),
]
for ax, col, title, key in panel_specs:
    for N in EXP3_N_VALUES:
        sub = df3[df3['N'] == N].sort_values('num_restarts')
        ax.plot(sub['num_restarts'], sub[col],
                marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
    ax.set_title(f'{title} wall time', fontsize=11, color=ALGO_COLORS[key])
    ax.set_xlabel('Restarts')
    ax.set_ylabel('Wall time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_wall_time.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_wall_time.png')


# Figure 10: Exp 3 — FLOP ratios vs Spectral
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Exp 3: FLOP ratio vs Spectral at each restart budget',
             fontsize=11, fontweight='bold')
ratio_specs = [
    (axes[0], 'flop_ratio',      'AMP / Spectral',   'AMP'),
    (axes[1], 'ampt_flop_ratio', 'AMP-T / Spectral', 'AMP-T'),
    (axes[2], 'iamp_flop_ratio', 'IAMP / Spectral',  'IAMP'),
]
for ax, col, title, key in ratio_specs:
    for N in EXP3_N_VALUES:
        sub = df3[df3['N'] == N].sort_values('num_restarts')
        ax.plot(sub['num_restarts'], sub[col],
                marker=N_MARKERS[N], lw=2, color=N_COLORS[N], label=f'N={N}')
    ax.axhline(1, color='k', lw=1, ls='--', alpha=0.4)
    ax.set_title(title, fontsize=11, color=ALGO_COLORS[key])
    ax.set_xlabel('Restarts')
    ax.set_ylabel('FLOP ratio')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_exp3_flop_ratio.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_exp3_flop_ratio.png')


# Figure 11: Energy trajectory
fig, ax = plt.subplots(figsize=(9, 5))
iters = np.arange(1, TRAJ_ITER + 1)
ax.plot(iters, gd_traj_eN, color=ALGO_COLORS['GD'],    lw=2, label='GD')
ax.plot(iters, amp_traj,   color=ALGO_COLORS['AMP'],   lw=2, label='AMP')
ax.plot(iters, ampt_traj,  color=ALGO_COLORS['AMP-T'], lw=2, label='AMP-T')
ax.plot(iters, iamp_traj,  color=ALGO_COLORS['IAMP'],  lw=2, label='IAMP')
ax.axhline(spec_eN_traj,   color=ALGO_COLORS['SPEC'],  lw=2, ls='--', label='Spectral')
ax.axhline(PARISI_VALUE,   color='black', lw=1.2, ls=':', label='Parisi')
ax.set_xlabel('Iteration')
ax.set_ylabel('Energy / N')
ax.set_title(f'Energy trajectory  (N={TRAJ_N}, single instance)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_trajectory.png')


# Figure 12: Empirical CDF of single-run energies
fig, ax = plt.subplots(figsize=(8, 5))
for algo, key, marker in [
    ('GD',       'GD',    'o'),
    ('AMP',      'AMP',   's'),
    ('AMP-T',    'AMP-T', 'P'),
    ('IAMP',     'IAMP',  'X'),
    ('Spectral', 'SPEC',  'D'),
]:
    vals = np.sort(cdf_energies[key])
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=ALGO_COLORS[key], lw=2,
            marker=marker, markersize=5, markevery=5, label=algo)
ax.axvline(PARISI_VALUE, color='black', lw=1.2, ls=':', label='Parisi')
ax.set_xlabel('Energy / N')
ax.set_ylabel('CDF')
ax.set_title(f'Single-run energy distribution  (N={CDF_N}, {CDF_SEEDS} seeds)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig_cdf.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_cdf.png')


# Figure 13: Log-log gap vs N with slope annotation
SLOPE_ITER = ITERATION_VALUES[len(ITERATION_VALUES) // 2]
sub_slope  = df1[df1['iterations'] == SLOPE_ITER].copy()

fig, ax = plt.subplots(figsize=(8, 5))
for algo, col, key, marker in [
    ('GD',       'gd_gap_pct',   'GD',    'o'),
    ('AMP',      'amp_gap_pct',  'AMP',   's'),
    ('AMP-T',    'ampt_gap_pct', 'AMP-T', 'P'),
    ('IAMP',     'iamp_gap_pct', 'IAMP',  'X'),
    ('Spectral', 'spec_gap_pct', 'SPEC',  'D'),
]:
    xs = sub_slope['N'].values.astype(float)
    ys = sub_slope[col].values.astype(float)
    ax.loglog(xs, ys, marker=marker, lw=2, color=ALGO_COLORS[key], label=algo)
    valid = ys > 0
    if valid.sum() >= 2:
        slope, _ = np.polyfit(np.log(xs[valid]), np.log(ys[valid]), 1)
        ax.annotate(
            f'{slope:.2f}',
            xy=(xs[valid][-1], ys[valid][-1]),
            xytext=(6, 0), textcoords='offset points',
            color=ALGO_COLORS[key], fontsize=9, va='center',
        )
ax.set_xlabel('N')
ax.set_ylabel('Gap to Parisi (%)')
ax.set_title(f'Gap scaling with N  ({SLOPE_ITER} iterations, slopes annotated)')
ax.legend()
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('fig_gap_loglog.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_gap_loglog.png')


# Figure 14: IAMP − AMP gap difference
fig, axes = plt.subplots(1, len(ITERATION_VALUES), figsize=(16, 4), sharey=True)
fig.suptitle('IAMP vs AMP gap difference  (negative = IAMP closer to Parisi)', fontsize=12)
for ax, it in zip(axes, ITERATION_VALUES):
    sub  = df1[df1['iterations'] == it].sort_values('N')
    diff = sub['iamp_gap_pct'].values - sub['amp_gap_pct'].values
    ax.plot(sub['N'], diff, 'o-', color=ALGO_COLORS['IAMP'], lw=2)
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax.set_title(f'{it} iterations')
    ax.set_xlabel('N')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('IAMP gap − AMP gap (pp)')
plt.tight_layout()
plt.savefig('fig_iamp_vs_amp_diff.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_iamp_vs_amp_diff.png')


# Figure 15: Win rate heatmap
winner_labels = ['GD', 'AMP', 'AMP-T', 'IAMP', 'SPEC']
win_matrix    = np.zeros((len(winner_labels), len(ITERATION_VALUES)))
for j, it in enumerate(ITERATION_VALUES):
    sub = df1[df1['iterations'] == it]
    vc  = sub['winner'].value_counts(normalize=True) * 100
    for i, w in enumerate(winner_labels):
        win_matrix[i, j] = vc.get(w, 0.0)

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(win_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
ax.set_xticks(range(len(ITERATION_VALUES)))
ax.set_xticklabels(ITERATION_VALUES)
ax.set_yticks(range(len(winner_labels)))
ax.set_yticklabels(winner_labels)
ax.set_xlabel('Iterations')
ax.set_title('Win rate — Exp 1 (%)')
for i in range(len(winner_labels)):
    for j in range(len(ITERATION_VALUES)):
        v = win_matrix[i, j]
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=11,
                color='white' if v > 55 else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('fig_win_rate_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_win_rate_heatmap.png')


# Figure 16: Crossover N table
crossover_rows = []
for it in ITERATION_VALUES:
    sub  = df1[df1['iterations'] == it].sort_values('N')
    ns   = sub['N'].values
    amp_cross = next(
        (ns[k] for k in range(len(ns))
         if sub['amp_gap_pct'].values[k] < sub['spec_gap_pct'].values[k]),
        f'>{ns[-1]}'
    )
    iamp_cross = next(
        (ns[k] for k in range(len(ns))
         if sub['iamp_gap_pct'].values[k] < sub['spec_gap_pct'].values[k]),
        f'>{ns[-1]}'
    )
    crossover_rows.append({'Iterations': it, 'AMP': amp_cross, 'IAMP': iamp_cross})

df_cross = pd.DataFrame(crossover_rows)
fig, ax = plt.subplots(figsize=(5, 2.2))
ax.axis('off')
tbl = ax.table(cellText=df_cross.values, colLabels=df_cross.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.8)
ax.set_title('First N where AMP / IAMP beats Spectral', pad=10)
plt.tight_layout()
plt.savefig('fig_crossover_table.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_crossover_table.png')


# Figure 17: Cost-of-quality table
QUALITY_THRESHOLD_PCT = 2.0

cq_rows = []
for N in EXP3_N_VALUES:
    sub = df3[df3['N'] == N].sort_values('num_restarts')
    row = {'N': N}
    for label, gap_col, flop_col in [
        ('AMP',      'amp_gap_pct',  'amp_flops'),
        ('AMP-T',    'ampt_gap_pct', 'ampt_flops'),
        ('IAMP',     'iamp_gap_pct', 'iamp_flops'),
        ('Spectral', 'spec_gap_pct', 'spec_flops'),
    ]:
        achieved = sub[sub[gap_col] <= QUALITY_THRESHOLD_PCT]
        row[label] = f"{achieved.iloc[0][flop_col]:.2e}" if len(achieved) else '—'
    cq_rows.append(row)

df_cq = pd.DataFrame(cq_rows)
fig, ax = plt.subplots(figsize=(9, 3))
ax.axis('off')
tbl = ax.table(cellText=df_cq.values, colLabels=df_cq.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 1.8)
ax.set_title(f'Min FLOPs to reach ≤{QUALITY_THRESHOLD_PCT}% gap to Parisi', pad=10)
plt.tight_layout()
plt.savefig('fig_cost_of_quality.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved fig_cost_of_quality.png')


print('\n✓ All figures saved.')
print('\nKey output files:')
print('  sk_exp1_single_run.csv         — Experiment 1 data')
print('  sk_exp2_fixed_budget.csv       — Experiment 2 data')
print('  sk_exp3_restart_sweep.csv      — Experiment 3 data')
print('  fig_exp1_gap_vs_N.png          — Exp 1: gap with uncertainty bands')
print('  fig_exp2_gap_vs_N.png          — Exp 2: gap under equal time budget')
print('  fig_exp2_restarts_vs_N.png     — Exp 2: restarts each algo completed')
print('  fig_exp1_vs_exp2_bar.png       — Side-by-side Exp 1 vs Exp 2')
print('  fig_exp1_variance_vs_N.png     — Exp 1: per-algorithm variance')
print('  fig_exp3_gap_vs_restarts.png   — Exp 3: gap vs restart count')
print('  fig_exp3_amp_advantage.png     — Exp 3: AMP/IAMP advantage over Spectral')
print('  fig_exp3_pareto.png            — Exp 3: quality vs FLOPs Pareto')
print('  fig_exp3_wall_time.png         — Exp 3: wall time vs restarts')
print('  fig_exp3_flop_ratio.png        — Exp 3: FLOP ratio vs Spectral')
print('  fig_trajectory.png             — Energy trajectory, single instance')
print('  fig_cdf.png                    — Single-run energy distribution')
print('  fig_gap_loglog.png             — Gap scaling with N, slopes annotated')
print('  fig_iamp_vs_amp_diff.png       — IAMP vs AMP gap difference')
print('  fig_win_rate_heatmap.png       — Win rate heatmap, Exp 1')
print('  fig_crossover_table.png        — First N where AMP/IAMP beats Spectral')
print('  fig_cost_of_quality.png        — Min FLOPs to reach gap threshold')