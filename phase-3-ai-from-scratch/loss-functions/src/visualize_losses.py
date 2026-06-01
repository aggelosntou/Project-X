"""
visualize_losses.py — Visualize each loss function and its gradient.

For each loss, compute and print:
  - Loss values for pred in [-3, 3], target = 0
  - Gradient values across the same range

Plots are saved if matplotlib is available.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from losses import MSELoss, MAELoss, HuberLoss, BinaryCrossEntropy


def compute_loss_curve(loss_class, pred_range: np.ndarray, target: float = 0.0,
                       loss_kwargs: dict = None) -> tuple:
    """
    Compute loss and gradient for scalar predictions.

    Returns: (losses, grads) arrays of same shape as pred_range
    """
    kwargs = loss_kwargs or {}
    losses = []
    grads  = []

    for p in pred_range:
        pred_arr   = np.array([p])
        target_arr = np.array([target])
        loss_obj   = loss_class(**kwargs)
        loss_val   = loss_obj.forward(pred_arr, target_arr)
        grad_val   = loss_obj.backward()[0]
        losses.append(float(loss_val))
        grads.append(float(grad_val))

    return np.array(losses), np.array(grads)


def print_loss_table(name: str, pred_range: np.ndarray,
                     losses: np.ndarray, grads: np.ndarray, n_rows: int = 11):
    """Print a compact table of pred, loss, grad values."""
    print(f"\n{'='*50}")
    print(f"{name}  (target = 0.0)")
    print(f"{'='*50}")
    print(f"  {'pred':>8} | {'loss':>10} | {'gradient':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}")

    step = max(1, len(pred_range) // n_rows)
    for i in range(0, len(pred_range), step):
        print(f"  {pred_range[i]:>8.2f} | {losses[i]:>10.4f} | {grads[i]:>10.4f}")


def visualize_all():
    pred_range = np.linspace(-3, 3, 61)
    target_val = 0.0

    print("=" * 50)
    print("LOSS FUNCTION VISUALIZATION")
    print("pred range: [-3, 3],  target = 0.0")
    print("=" * 50)

    # ── MSE ──────────────────────────────────────────────────────────────
    losses_mse, grads_mse = compute_loss_curve(MSELoss, pred_range)
    print_loss_table("MSE Loss:  L = (pred - target)²  (mean over N=1)",
                     pred_range, losses_mse, grads_mse)
    print(f"  Key: parabolic curve, gradient grows linearly with error")
    print(f"  At pred=2.0: loss={losses_mse[np.argmin(np.abs(pred_range-2)):.0f]:.4f}, "
          f"grad={grads_mse[np.argmin(np.abs(pred_range-2)):.0f]:.4f}")

    # ── MAE ──────────────────────────────────────────────────────────────
    losses_mae, grads_mae = compute_loss_curve(MAELoss, pred_range)
    print_loss_table("MAE Loss:  L = |pred - target|",
                     pred_range, losses_mae, grads_mae)
    print(f"  Key: V-shaped, gradient is constant ±1 (robust to outliers)")
    print(f"  At pred=2.0: loss={losses_mae[np.argmin(np.abs(pred_range-2)):.0f]:.4f}, "
          f"grad={grads_mae[np.argmin(np.abs(pred_range-2)):.0f]:.4f}")

    # ── Huber ─────────────────────────────────────────────────────────────
    for delta in [0.5, 1.0, 2.0]:
        losses_h, grads_h = compute_loss_curve(HuberLoss, pred_range,
                                               loss_kwargs={'delta': delta})
        print_loss_table(f"Huber Loss (δ={delta}):  quadratic for |e|<δ, linear otherwise",
                         pred_range, losses_h, grads_h)
    print(f"  Key: MSE near zero, MAE for outliers. δ controls transition point.")

    # ── Comparison table (key error values) ──────────────────────────────
    print(f"\n{'='*65}")
    print("COMPARISON AT KEY ERROR VALUES (target=0)")
    print(f"{'='*65}")
    print(f"  {'pred':>6} | {'MSE loss':>10} | {'MAE loss':>10} | {'Huber(δ=1)':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    key_preds = [0.0, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0]
    for kp in key_preds:
        idx   = np.argmin(np.abs(pred_range - kp))
        mse_v = losses_mse[idx]
        mae_v = losses_mae[idx]
        hub_v = compute_loss_curve(HuberLoss, np.array([kp]),
                                   loss_kwargs={'delta': 1.0})[0][0]
        print(f"  {kp:>6.1f} | {mse_v:>10.4f} | {mae_v:>10.4f} | {hub_v:>10.4f}")

    print(f"\n  Observation: at pred=3.0, MSE={losses_mse[np.argmin(np.abs(pred_range-3)):.0f]:.4f} "
          f"vs MAE={losses_mae[np.argmin(np.abs(pred_range-3)):.0f]:.4f} "
          f"(MSE is {losses_mse[np.argmin(np.abs(pred_range-3)):.0f]/losses_mae[np.argmin(np.abs(pred_range-3)):.0f]:.1f}x larger)")

    # ── BCE visualization ─────────────────────────────────────────────────
    prob_range = np.linspace(0.01, 0.99, 51)
    bce_y1_losses = []
    bce_y1_grads  = []
    bce_y0_losses = []
    bce_y0_grads  = []

    for p in prob_range:
        pred_arr = np.array([p])
        # y=1 case
        bce = BinaryCrossEntropy()
        bce_y1_losses.append(float(bce.forward(pred_arr, np.array([1.0]))))
        bce_y1_grads.append(float(bce.backward()[0]))
        # y=0 case
        bce = BinaryCrossEntropy()
        bce_y0_losses.append(float(bce.forward(pred_arr, np.array([0.0]))))
        bce_y0_grads.append(float(bce.backward()[0]))

    print(f"\n{'='*55}")
    print("BINARY CROSS-ENTROPY:  L = -[y*log(p) + (1-y)*log(1-p)]")
    print(f"{'='*55}")
    print(f"  {'prob p':>8} | {'L (y=1)':>10} | {'L (y=0)':>10} | {'dL/dp (y=1)':>12}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    step = max(1, len(prob_range) // 12)
    for i in range(0, len(prob_range), step):
        print(f"  {prob_range[i]:>8.2f} | {bce_y1_losses[i]:>10.4f} | "
              f"{bce_y0_losses[i]:>10.4f} | {bce_y1_grads[i]:>12.4f}")

    print(f"\n  Key: when y=1 and p→0, loss→∞ (large gradient pulls p toward 1)")
    print(f"       when y=1 and p→1, loss→0 (no gradient needed)")

    # ── Try matplotlib plots ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()

        # Plot 1: Loss curves comparison
        ax = axes[0]
        ax.plot(pred_range, losses_mse, label='MSE', linewidth=2)
        ax.plot(pred_range, losses_mae, label='MAE', linewidth=2)
        for delta in [0.5, 1.0, 2.0]:
            l, _ = compute_loss_curve(HuberLoss, pred_range, loss_kwargs={'delta': delta})
            ax.plot(pred_range, l, label=f'Huber δ={delta}', linewidth=1.5, linestyle='--')
        ax.set_title('Loss Functions (target=0)'); ax.set_xlabel('pred'); ax.set_ylabel('loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 5)

        # Plot 2: Gradient comparison
        ax = axes[1]
        ax.plot(pred_range, grads_mse, label='MSE', linewidth=2)
        ax.plot(pred_range, grads_mae, label='MAE', linewidth=2)
        _, g_h = compute_loss_curve(HuberLoss, pred_range, loss_kwargs={'delta': 1.0})
        ax.plot(pred_range, g_h, label='Huber δ=1.0', linewidth=2, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title('Gradients (target=0)'); ax.set_xlabel('pred'); ax.set_ylabel('dL/dpred')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Plot 3: BCE
        ax = axes[2]
        ax.plot(prob_range, bce_y1_losses, label='L (y=1)', linewidth=2)
        ax.plot(prob_range, bce_y0_losses, label='L (y=0)', linewidth=2)
        ax.set_title('Binary Cross-Entropy'); ax.set_xlabel('p (predicted prob)')
        ax.set_ylabel('loss'); ax.legend(); ax.grid(True, alpha=0.3)

        # Plot 4: BCE gradient
        ax = axes[3]
        ax.plot(prob_range, bce_y1_grads, label='dL/dp (y=1)', linewidth=2)
        ax.plot(prob_range, bce_y0_grads, label='dL/dp (y=0)', linewidth=2)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title('BCE Gradient'); ax.set_xlabel('p'); ax.set_ylabel('dL/dp')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Plot 5: Huber vs MSE at large errors
        ax = axes[4]
        big_range = np.linspace(-10, 10, 200)
        l_mse, _ = compute_loss_curve(MSELoss, big_range)
        l_mae, _ = compute_loss_curve(MAELoss, big_range)
        l_hub, _ = compute_loss_curve(HuberLoss, big_range, loss_kwargs={'delta': 1.0})
        ax.plot(big_range, l_mse, label='MSE', linewidth=2)
        ax.plot(big_range, l_mae, label='MAE', linewidth=2)
        ax.plot(big_range, l_hub, label='Huber δ=1', linewidth=2, linestyle='--')
        ax.set_title('Outlier Robustness (large range)'); ax.set_xlabel('pred')
        ax.set_ylabel('loss'); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 30)

        # Plot 6: Huber loss shape at different deltas
        ax = axes[5]
        small_range = np.linspace(-3, 3, 200)
        for delta, ls in [(0.25, ':'), (0.5, '--'), (1.0, '-'), (2.0, '-.')]:
            l, _ = compute_loss_curve(HuberLoss, small_range, loss_kwargs={'delta': delta})
            ax.plot(small_range, l, label=f'δ={delta}', linewidth=2, linestyle=ls)
        ax.set_title('Huber: Effect of δ'); ax.set_xlabel('pred - target')
        ax.set_ylabel('loss'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('Loss Functions Overview', fontsize=14)
        plt.tight_layout()

        out_path = os.path.join(os.path.dirname(__file__), '..', 'loss_functions.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"\nSaved plot: {out_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping plots")


if __name__ == "__main__":
    np.random.seed(42)
    visualize_all()
