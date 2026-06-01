"""
compare_optimizers.py — Compare all optimizers on standard test functions.

For each optimizer and each test function:
  - Run for up to 10000 steps
  - Record trajectory, loss at each step
  - Report: steps to converge, final loss, converged (yes/no)
  - Plot trajectories if matplotlib is available
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from test_functions import rosenbrock, beale, himmelblau, rastrigin
from optimizers import SGD, SGDMomentum, Nesterov, RMSProp, Adam, AdamW

CONVERGENCE_THRESHOLD = 0.01
MAX_STEPS = 10000


def run_optimizer(optimizer, fn, x0: float, y0: float,
                  max_steps: int = MAX_STEPS,
                  threshold: float = CONVERGENCE_THRESHOLD,
                  lr_override: float = None):
    """
    Run optimizer on a 2D function from starting point (x0, y0).

    Returns:
        trajectory:    list of (x, y) at each step
        losses:        list of f values
        steps_to_conv: steps until loss < threshold (None if not converged)
    """
    if lr_override is not None:
        optimizer.lr = lr_override

    params = np.array([x0, y0])
    trajectory = [(float(params[0]), float(params[1]))]
    losses = []
    steps_to_conv = None

    for step in range(max_steps):
        f, g = fn(params[0], params[1])
        losses.append(float(f))

        if steps_to_conv is None and f < threshold:
            steps_to_conv = step

        # Check for divergence
        if not np.isfinite(f) or np.linalg.norm(params) > 1e6:
            break

        optimizer.step([(params, g)])
        trajectory.append((float(params[0]), float(params[1])))

    return trajectory, losses, steps_to_conv


def format_row(name, fn_name, steps_to_conv, final_loss, converged):
    """Format a comparison table row."""
    steps_str = str(steps_to_conv) if steps_to_conv is not None else "N/A"
    conv_str  = "YES" if converged else "NO"
    return (f"  {name:<15} | {fn_name:<12} | {steps_str:>12} | "
            f"{final_loss:>12.6f} | {conv_str:>9}")


def compare_all():
    """Run all optimizers on all test functions and print comparison table."""

    # Optimizer configs: (name, class, kwargs, lr_for_each_fn)
    optimizer_configs = [
        ("SGD",         SGD,         {'lr': 0.001},        {'rosenbrock': 0.0001, 'beale': 0.0001, 'himmelblau': 0.001, 'rastrigin': 0.001}),
        ("SGDMomentum", SGDMomentum, {'lr': 0.001},        {'rosenbrock': 0.0001, 'beale': 0.0001, 'himmelblau': 0.001, 'rastrigin': 0.001}),
        ("Nesterov",    Nesterov,    {'lr': 0.001},        {'rosenbrock': 0.0001, 'beale': 0.0001, 'himmelblau': 0.001, 'rastrigin': 0.001}),
        ("RMSProp",     RMSProp,     {'lr': 0.01},         {'rosenbrock': 0.005,  'beale': 0.005,  'himmelblau': 0.01,  'rastrigin': 0.01}),
        ("Adam",        Adam,        {'lr': 0.01},         {'rosenbrock': 0.005,  'beale': 0.005,  'himmelblau': 0.01,  'rastrigin': 0.01}),
        ("AdamW",       AdamW,       {'lr': 0.01, 'weight_decay': 0.001},
                                                           {'rosenbrock': 0.005,  'beale': 0.005,  'himmelblau': 0.01,  'rastrigin': 0.01}),
    ]

    test_cases = [
        ('rosenbrock', rosenbrock, -1.0, 1.0),
        ('beale',      beale,       1.0, 1.0),
        ('himmelblau', himmelblau,  0.0, 0.0),
        ('rastrigin',  rastrigin,   2.0, 2.0),
    ]

    print("=" * 70)
    print("OPTIMIZER COMPARISON ON TEST FUNCTIONS")
    print(f"Convergence threshold: f < {CONVERGENCE_THRESHOLD}")
    print(f"Max steps: {MAX_STEPS}")
    print("=" * 70)

    all_results = {}

    for fn_name, fn, x0, y0 in test_cases:
        print(f"\n--- {fn_name.upper()} (start: ({x0}, {y0})) ---")
        print(f"  {'Optimizer':<15} | {'Function':<12} | {'Steps→conv':>12} | "
              f"{'Final loss':>12} | {'Converged':>9}")
        print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}")

        fn_results = {}

        for name, cls, kwargs, lr_map in optimizer_configs:
            lr = lr_map.get(fn_name, kwargs['lr'])
            opt_kwargs = {**kwargs, 'lr': lr}
            opt = cls(**opt_kwargs)

            traj, losses, steps_conv = run_optimizer(
                opt, fn, x0, y0, MAX_STEPS, CONVERGENCE_THRESHOLD
            )

            final_loss = losses[-1] if losses else float('inf')
            converged  = steps_conv is not None or final_loss < CONVERGENCE_THRESHOLD

            print(format_row(name, fn_name, steps_conv, final_loss, converged))

            fn_results[name] = {
                'trajectory': traj,
                'losses':     losses,
                'steps_conv': steps_conv,
                'final_loss': final_loss,
                'converged':  converged,
            }

        all_results[fn_name] = fn_results

    # ── Summary across all functions ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Average final loss across all test functions")
    print("=" * 70)
    optimizer_names = [cfg[0] for cfg in optimizer_configs]
    fn_names        = [tc[0] for tc in test_cases]

    print(f"  {'Optimizer':<15} | {'Avg Final Loss':>14} | {'Conv Rate':>9}")
    print(f"  {'-'*15}-+-{'-'*14}-+-{'-'*9}")

    for name in optimizer_names:
        losses    = [all_results[fn][name]['final_loss'] for fn in fn_names]
        convs     = [all_results[fn][name]['converged']  for fn in fn_names]
        avg_loss  = np.mean([l for l in losses if np.isfinite(l)])
        conv_rate = sum(convs) / len(convs)
        print(f"  {name:<15} | {avg_loss:>14.6f} | {conv_rate:>8.0%}")

    # ── Trajectory plot (if matplotlib available) ─────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        print("\nGenerating trajectory plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()

        plot_configs = [
            ('rosenbrock', rosenbrock, (-2, 2), (-1, 3),    "Rosenbrock (min at (1,1))"),
            ('beale',      beale,      (-4.5, 4.5), (-4.5, 4.5), "Beale (min at (3, 0.5))"),
            ('himmelblau', himmelblau, (-5, 5), (-5, 5),    "Himmelblau (4 minima)"),
            ('rastrigin',  rastrigin,  (-3, 3), (-3, 3),    "Rastrigin (many minima)"),
        ]

        colors = list(mcolors.TABLEAU_COLORS.values())[:len(optimizer_names)]

        for ax_idx, (fn_name, fn, x_rng, y_rng, title) in enumerate(plot_configs):
            ax = axes[ax_idx]

            # Background contour
            xx = np.linspace(x_rng[0], x_rng[1], 200)
            yy = np.linspace(y_rng[0], y_rng[1], 200)
            XX, YY = np.meshgrid(xx, yy)
            ZZ = np.vectorize(lambda x, y: fn(x, y)[0])(XX, YY)
            ZZ_log = np.log1p(np.abs(ZZ))
            ax.contourf(XX, YY, ZZ_log, levels=30, cmap='viridis', alpha=0.6)
            ax.contour(XX,  YY, ZZ_log, levels=30, colors='white', alpha=0.3, linewidths=0.5)

            # Trajectories
            for i, name in enumerate(optimizer_names):
                traj = all_results[fn_name][name]['trajectory']
                if len(traj) < 2:
                    continue
                xs = [t[0] for t in traj[:500]]   # first 500 steps
                ys = [t[1] for t in traj[:500]]
                ax.plot(xs, ys, color=colors[i], linewidth=1, label=name, alpha=0.8)
                ax.plot(xs[0], ys[0], 'o', color=colors[i], markersize=5)

            ax.set_title(title, fontsize=11)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_xlim(x_rng); ax.set_ylim(y_rng)
            if ax_idx == 0:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        plt.suptitle("Optimizer Trajectories on Test Functions", fontsize=13)
        plt.tight_layout()

        out_path = os.path.join(os.path.dirname(__file__), '..', 'optimizer_trajectories.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"  Saved: {out_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping trajectory plots")

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    compare_all()
