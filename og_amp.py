"""
SK Model Benchmarking Suite
Compares Gradient Descent and AMP for finding ground states of the
Sherrington-Kirkpatrick spin glass and related structured models.

Hamiltonian convention:  H(sigma) = -0.5 * sigma^T J sigma
Goal: minimize H  (equivalently, maximize sigma^T J sigma)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Unified return type for every algorithm."""
    name: str
    final_spins: np.ndarray
    final_energy: float                 # raw energy H(sigma_final)
    final_energy_per_spin: float        # H/N, comparable across sizes
    wall_time: float                    # total seconds
    # trajectory: list of (elapsed_seconds, energy_per_spin) tuples
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    flop_total: int = 0
    flop_breakdown: Dict[str, int] = field(default_factory=dict)


def dense_matvec_flops(n: int) -> int:
    """Approximate FLOPs for dense matrix-vector multiply: y = A x."""
    return n * (2 * n - 1)


def dot_flops(n: int) -> int:
    """Approximate FLOPs for dot product."""
    return 2 * n - 1


def add_flops(breakdown: Dict[str, int], key: str, value: int) -> None:
    breakdown[key] = breakdown.get(key, 0) + int(value)


# ---------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------

def energy(sigma: np.ndarray, J: np.ndarray) -> float:
    """H = -0.5 * sigma^T J sigma."""
    return float(-0.5 * sigma @ (J @ sigma))


def energy_per_spin(sigma: np.ndarray, J: np.ndarray) -> float:
    return energy(sigma, J) / len(sigma)


# ---------------------------------------------------------------------
# Matrix generators
# ---------------------------------------------------------------------

def generate_sk_matrix(N: int, seed: int = None) -> Tuple[np.ndarray, float]:
    """Standard Sherrington-Kirkpatrick: GOE-like symmetric Gaussian matrix."""
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N))
    J = (J + J.T) / np.sqrt(2.0)
    np.fill_diagonal(J, 0.0)
    # Parisi ground state energy density (n -> infinity)
    parisi = -0.7633
    return J, parisi


def generate_structured_matrix(N: int, seed: int = None) -> Tuple[np.ndarray, float]:
    """Rank-1 planted matrix. Single perfect ground state at the planted pattern."""
    rng = np.random.default_rng(seed)
    pattern = rng.choice([-1.0, 1.0], size=N)
    J = np.outer(pattern, pattern) / N
    np.fill_diagonal(J, 0.0)
    # Energy of the planted pattern: H = -0.5 * (N - 1) ~= -0.5 N
    floor_per_spin = -0.5
    return J, floor_per_spin


def generate_hopfield_matrix(N: int, num_patterns: int = 10,
                             seed: int = None) -> Tuple[np.ndarray, float]:
    """
    Hopfield-style associative memory: K planted patterns superposed.
    The retrieval phase exists for alpha = K/N below ~0.138.
    Below that capacity, the planted patterns are local minima with
    energy density approximately -0.5 (per pattern, ignoring crosstalk).
    """
    rng = np.random.default_rng(seed)
    J = np.zeros((N, N))
    for _ in range(num_patterns):
        p = rng.choice([-1.0, 1.0], size=N)
        J += np.outer(p, p)
    J /= N
    np.fill_diagonal(J, 0.0)
    # Approximate retrieval energy density. For alpha << 0.138 this is ~-0.5.
    # We return -0.5 as the target floor; gap to it is meaningful below capacity.
    floor_per_spin = -0.5
    return J, floor_per_spin


# ---------------------------------------------------------------------
# Algorithm 1: Projected Gradient Descent
# ---------------------------------------------------------------------

def run_gradient_descent(J: np.ndarray, num_iterations: int = 200,
                         learning_rate: float = 0.1,
                         seed: int = None) -> BenchmarkResult:
    """
    Projected gradient descent on H = -0.5 sigma^T J sigma over [-1, 1]^N.
    Gradient: dH/dsigma = -J sigma.
    To minimize H we step in -gradient direction = +J sigma.
    """
    rng = np.random.default_rng(seed)
    N = J.shape[0]
    sigma = rng.uniform(-1.0, 1.0, size=N)
    flops: Dict[str, int] = {}

    trajectory = []
    t_start = time.perf_counter()

    for _ in range(num_iterations):
        # gradient of H
        grad = -(J @ sigma)
        add_flops(flops, "matvec_grad", dense_matvec_flops(N))
        add_flops(flops, "grad_negation", N)
        # step opposite the gradient to MINIMIZE H
        sigma = sigma - learning_rate * grad
        add_flops(flops, "update_step", 2 * N)
        sigma = np.clip(sigma, -1.0, 1.0)
        add_flops(flops, "clip_ops", N)

        elapsed = time.perf_counter() - t_start
        e_iter = energy_per_spin(sigma, J)
        add_flops(flops, "trajectory_energy_matvec", dense_matvec_flops(N))
        add_flops(flops, "trajectory_energy_dot", dot_flops(N))
        add_flops(flops, "trajectory_energy_scale", 1)
        trajectory.append((elapsed, e_iter))

    wall_time = time.perf_counter() - t_start
    final_e = energy(sigma, J)
    add_flops(flops, "final_energy_matvec", dense_matvec_flops(N))
    add_flops(flops, "final_energy_dot", dot_flops(N))
    add_flops(flops, "final_energy_scale", 1)

    total_flops = sum(flops.values())

    return BenchmarkResult(
        name="Gradient Descent",
        final_spins=sigma,
        final_energy=final_e,
        final_energy_per_spin=final_e / N,
        wall_time=wall_time,
        trajectory=trajectory,
        flop_total=total_flops,
        flop_breakdown=flops,
    )


# ---------------------------------------------------------------------
# Algorithm 2: Damped AMP with cooling schedule + greedy quench
# ---------------------------------------------------------------------

def amp_iterate(J: np.ndarray, m_init: np.ndarray,
                num_iterations: int = 1000,
                damping: float = 0.7,
                beta_max: float = 0.99) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Run AMP with a temperature-cooling schedule from beta=0.1 to beta_max.
    Stops below the algorithmic threshold (beta=1) where TAP becomes ill-posed.
    Returns continuous magnetizations m in [-1, 1]^N (NOT yet rounded).
    """
    N = J.shape[0]
    m = np.copy(m_init)
    m_old = np.zeros(N)
    h = np.zeros(N)
    betas = np.linspace(0.1, beta_max, num_iterations)
    flops: Dict[str, int] = {}

    for beta in betas:
        # Onsager correction: cancels self-feedback
        onsager_coef = np.mean(1.0 - m**2)
        add_flops(flops, "onsager_square", N)
        add_flops(flops, "onsager_subtract_one", N)
        add_flops(flops, "onsager_mean", N)

        h_target = J @ m - beta * onsager_coef * m_old
        add_flops(flops, "matvec_Jm", dense_matvec_flops(N))
        add_flops(flops, "onsager_scale_vector", 2 * N)
        add_flops(flops, "h_target_subtract", N)

        # field damping for numerical stability
        h = damping * h + (1.0 - damping) * h_target
        add_flops(flops, "damping_mix", 3 * N)
        m_old = np.copy(m)
        m = np.tanh(beta * h)
        add_flops(flops, "tanh_update", 2 * N)

    return m, flops


def greedy_quench(sigma: np.ndarray, J: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Local search: flip the most-frustrated spin until none are frustrated.
    A spin is frustrated when sigma_i * (J sigma)_i < 0 (i.e. the local
    field disagrees with its current sign).
    """
    sigma_opt = np.sign(sigma).astype(float)
    # handle zero spins from sign(0)
    sigma_opt[sigma_opt == 0] = 1.0
    N = len(sigma_opt)
    flops: Dict[str, int] = {}

    while True:
        local_fields = J @ sigma_opt
        add_flops(flops, "quench_matvec", dense_matvec_flops(N))
        frustration = sigma_opt * local_fields
        add_flops(flops, "quench_frustration", N)
        idx = int(np.argmin(frustration))
        add_flops(flops, "quench_argmin", N - 1)
        if frustration[idx] >= 0:
            break
        sigma_opt[idx] *= -1.0
        add_flops(flops, "quench_flip", 1)

    return sigma_opt, flops


def get_orthogonal_starts(num_starts: int, N: int,
                          scale: float = 1e-3,
                          seed: int = None) -> np.ndarray:
    """Return num_starts orthogonal vectors in R^N, each scaled to ||.|| ~ scale."""
    assert num_starts <= N, f"Cannot have {num_starts} orthogonal vectors in R^{N}"
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, num_starts))
    Q, _ = np.linalg.qr(A)
    return Q.T * scale


def run_amp_multistart(J: np.ndarray, num_restarts: int = 25,
                       amp_iterations: int = 1000,
                       damping: float = 0.7,
                       seed: int = None,
                       verbose: bool = False) -> BenchmarkResult:
    """
    Multi-start AMP: run AMP from num_restarts orthogonal initializations,
    quench each result, return the best.
    Trajectory records (cumulative_time, best_energy_per_spin_so_far) after each restart.
    """
    N = J.shape[0]
    starts = get_orthogonal_starts(num_restarts, N, seed=seed)

    best_spins = None
    best_e = np.inf
    trajectory = []
    flops: Dict[str, int] = {}

    t_start = time.perf_counter()

    for i in range(num_restarts):
        m_continuous, amp_flops = amp_iterate(J, starts[i],
                                              num_iterations=amp_iterations,
                                              damping=damping)
        for k, v in amp_flops.items():
            add_flops(flops, k, v)

        spins, quench_flops = greedy_quench(m_continuous, J)
        for k, v in quench_flops.items():
            add_flops(flops, k, v)

        e = energy(spins, J)
        add_flops(flops, "restart_energy_matvec", dense_matvec_flops(N))
        add_flops(flops, "restart_energy_dot", dot_flops(N))
        add_flops(flops, "restart_energy_scale", 1)

        if e < best_e:
            best_e = e
            best_spins = spins

        elapsed = time.perf_counter() - t_start
        trajectory.append((elapsed, best_e / N))

        if verbose:
            gap = best_e / N - (-0.7633)
            print(f"  Restart {i+1:2d}/{num_restarts}: "
                  f"e/N = {e/N:+.4f} | best so far: {best_e/N:+.4f}")

    wall_time = time.perf_counter() - t_start
    total_flops = sum(flops.values())

    return BenchmarkResult(
        name="AMP (multi-start)",
        final_spins=best_spins,
        final_energy=best_e,
        final_energy_per_spin=best_e / N,
        wall_time=wall_time,
        trajectory=trajectory,
        flop_total=total_flops,
        flop_breakdown=flops,
    )


# ---------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------

def run_benchmark(N: int = 1000, mode: str = "RANDOM",
                  num_restarts: int = 15, gd_iterations: int = 300,
                  amp_iterations: int = 800, seed: int = 0,
                  verbose: bool = True):
    """Run all algorithms on a single problem instance and report."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SK Benchmark | mode={mode} | N={N} | seed={seed}")
        print(f"{'='*60}")

    # --- generate problem ---
    t0 = time.perf_counter()
    if mode == "RANDOM":
        J, target_per_spin = generate_sk_matrix(N, seed=seed)
    elif mode == "STRUCTURED":
        J, target_per_spin = generate_structured_matrix(N, seed=seed)
    elif mode == "HOPFIELD":
        J, target_per_spin = generate_hopfield_matrix(N, num_patterns=10, seed=seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    matrix_time = time.perf_counter() - t0

    if verbose:
        print(f"  Matrix generation: {matrix_time:.2f}s")
        print(f"  Target energy density: {target_per_spin:+.4f}")
        print()

    # --- run gradient descent ---
    if verbose:
        print("  Running gradient descent...")
    gd_result = run_gradient_descent(J, num_iterations=gd_iterations,
                                     learning_rate=0.1, seed=seed)

    # --- run AMP multi-start ---
    if verbose:
        print(f"  Running AMP with {num_restarts} restarts...")
    amp_result = run_amp_multistart(J, num_restarts=num_restarts,
                                    amp_iterations=amp_iterations,
                                    seed=seed, verbose=False)

    # --- report ---
    if verbose:
        print(f"\n  {'Algorithm':<22} {'e/N':>10} {'gap':>10} {'time(s)':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        for r in [gd_result, amp_result]:
            gap = r.final_energy_per_spin - target_per_spin
            print(f"  {r.name:<22} {r.final_energy_per_spin:>+10.4f} "
                  f"{gap:>+10.4f} {r.wall_time:>10.2f}")
        print()

        print(f"  {'Algorithm':<22} {'FLOPs':>18} {'GFLOPs':>12}")
        print(f"  {'-'*22} {'-'*18} {'-'*12}")
        for r in [gd_result, amp_result]:
            print(f"  {r.name:<22} {r.flop_total:>18,d} {r.flop_total / 1e9:>12.4f}")
        print()

        amp_iter_matvec = amp_result.flop_breakdown.get("matvec_Jm", 0)
        amp_all_matvec = sum(v for k, v in amp_result.flop_breakdown.items()
                     if "matvec" in k)
        amp_extra = amp_result.flop_total - amp_all_matvec
        extra_pct = 100.0 * amp_extra / max(1, amp_result.flop_total)
        print("  AMP FLOP breakdown:")
        print(f"    AMP-iterate Jm matvec:{amp_iter_matvec:,}")
        print(f"    all matvec total:     {amp_all_matvec:,}")
        print(f"    non-matvec total:     {amp_extra:,} ({extra_pct:.2f}% of AMP total)")

        for key in sorted(amp_result.flop_breakdown):
            if key == "matvec_Jm":
                continue
            print(f"    {key:<22} {amp_result.flop_breakdown[key]:>12,}")
        print()

    return {
        "J": J,
        "target_per_spin": target_per_spin,
        "gd": gd_result,
        "amp": amp_result,
        "N": N,
        "mode": mode,
    }


def plot_trajectories(results: dict, save_path: str = None):
    """Plot energy/N vs wall time for both algorithms on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    gd = results["gd"]
    amp = results["amp"]
    target = results["target_per_spin"]

    gd_t, gd_e = zip(*gd.trajectory)
    amp_t, amp_e = zip(*amp.trajectory)

    ax.plot(gd_t, gd_e, color="#D85A30", linestyle="--",
            linewidth=1.8, label="Gradient Descent", marker="o", markersize=3)
    ax.plot(amp_t, amp_e, color="#1D9E75",
            linewidth=1.8, label="AMP (best so far)", marker="s", markersize=4)
    ax.axhline(target, color="gray", linestyle=":", linewidth=1.2,
               label=f"Target ({target:+.4f})")

    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("energy per spin")
    ax.set_title(f"SK benchmark | mode={results['mode']} | N={results['N']}")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # default: a tractable size that still shows the AMP-vs-GD gap clearly
    results = run_benchmark(N=1000, mode="RANDOM",
                            num_restarts=15,
                            gd_iterations=300,
                            amp_iterations=800,
                            seed=42, verbose=True)

    plot_trajectories(results, save_path="sk_benchmark_random.png")

    # uncomment to also test the planted models:
    # res_struct = run_benchmark(N=1000, mode="STRUCTURED", seed=42)
    # plot_trajectories(res_struct, save_path="sk_benchmark_structured.png")
    # res_hop = run_benchmark(N=1000, mode="HOPFIELD", seed=42)
    # plot_trajectories(res_hop, save_path="sk_benchmark_hopfield.png")