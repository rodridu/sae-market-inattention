"""
Real-time training loss monitor

Usage:
  python monitor_training_loss.py data/sae_anthropic_sentences_item1_v2

This will:
1. Display current loss statistics
2. Plot loss curves
3. Detect if loss has plateaued (elbow point)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import time

def analyze_loss_convergence(loss_values, window=10):
    """
    Detect if loss has plateaued

    Returns:
        converged (bool): True if loss appears to have converged
        improvement_rate (float): Recent rate of improvement (negative = getting worse)
    """
    if len(loss_values) < window * 2:
        return False, None

    # Compare recent window to previous window
    recent = np.mean(loss_values[-window:])
    previous = np.mean(loss_values[-2*window:-window])

    improvement = previous - recent
    improvement_rate = improvement / previous if previous > 0 else 0

    # Converged if improvement is less than 0.1%
    converged = improvement_rate < 0.001

    return converged, improvement_rate

def plot_loss_curves(df, save_path=None):
    """Plot training loss curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(df['step'], df['loss'], linewidth=0.5, alpha=0.7)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (MSE + Sparsity)')
    axes[0, 0].grid(True, alpha=0.3)

    # MSE
    axes[0, 1].plot(df['step'], df['mse'], linewidth=0.5, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Reconstruction Error (MSE)')
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity
    axes[1, 0].plot(df['step'], df['sparsity'], linewidth=0.5, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Sparsity Penalty')
    axes[1, 0].set_title('Sparsity Penalty (L1)')
    axes[1, 0].grid(True, alpha=0.3)

    # Active features
    axes[1, 1].plot(df['step'], df['active_features'], linewidth=0.5, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Active Features')
    axes[1, 1].set_title('Mean Active Features per Sample')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

def monitor_training(save_dir, refresh_interval=30):
    """
    Monitor training progress in real-time

    Args:
        save_dir: Directory containing training_loss.csv
        refresh_interval: Seconds between updates (None = one-time check)
    """
    save_dir = Path(save_dir)
    loss_file = save_dir / "training_loss.csv"

    if not loss_file.exists():
        print(f"Error: Loss file not found at {loss_file}")
        print("Make sure training has started and is using 03b_sae_anthropic_v2.py")
        return

    print("="*80)
    print(f"MONITORING: {save_dir}")
    print("="*80)

    if refresh_interval:
        print(f"Refreshing every {refresh_interval} seconds. Press Ctrl+C to stop.\n")

        try:
            while True:
                display_loss_stats(loss_file, save_dir)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        display_loss_stats(loss_file, save_dir)

def display_loss_stats(loss_file, save_dir):
    """Display current loss statistics"""
    df = pd.read_csv(loss_file)

    if len(df) == 0:
        print("No data yet...")
        return

    latest = df.iloc[-1]

    # Convergence analysis
    converged, improvement_rate = analyze_loss_convergence(df['loss'].values, window=10)

    print(f"\n{'-'*80}")
    print(f"Last updated: {pd.Timestamp.now()}")
    print(f"{'-'*80}")
    print(f"Step:             {latest['step']:>10,}")
    print(f"Loss:             {latest['loss']:>10.6f}")
    print(f"  MSE:            {latest['mse']:>10.6f}")
    print(f"  Sparsity:       {latest['sparsity']:>10.6f}")
    print(f"Active features:  {latest['active_features']:>10.1f}")
    print(f"Lambda:           {latest['lambda']:>10.4f}")

    if improvement_rate is not None:
        print(f"\nImprovement rate: {improvement_rate*100:>10.4f}%")
        if converged:
            print("Status:           CONVERGED (elbow point reached)")
            print("                  Consider stopping training to save resources!")
        else:
            print("Status:           IMPROVING")

    # Show recent trend (last 10 points)
    if len(df) >= 10:
        recent_losses = df['loss'].tail(10).values
        trend = "DECREASING" if recent_losses[-1] < recent_losses[0] else "INCREASING/FLAT"
        print(f"Recent trend:     {trend}")
        print(f"  Last 10 avg:    {np.mean(recent_losses):.6f}")
        print(f"  Std dev:        {np.std(recent_losses):.6f}")

    print(f"{'-'*80}")

    # Generate plot every time
    plot_path = save_dir / "loss_curve_latest.png"
    plot_loss_curves(df, save_path=plot_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_training_loss.py <save_dir> [--once]")
        print("\nExample:")
        print("  python monitor_training_loss.py data/sae_anthropic_sentences_item1_v2")
        print("  python monitor_training_loss.py data/sae_anthropic_sentences_item1_v2 --once")
        sys.exit(1)

    save_dir = sys.argv[1]
    once = "--once" in sys.argv

    monitor_training(save_dir, refresh_interval=None if once else 30)
