"""
Post-hoc training analytics with publication-quality plots.

Reads logs/metrics.csv and generates:
  1. Reward curve with rolling mean and std bands
  2. Crash rate progression
  3. Episode length over time
  4. Difficulty / curriculum level progression
  5. Center distance and laps
  6. Combined summary dashboard
  7. Statistical summary text report

Usage:
    python analytics.py
    python analytics.py --csv logs/metrics.csv --output-dir figures/
    python analytics.py --window 50 --dpi 200
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_metrics(csv_path: str) -> dict[str, np.ndarray]:
    """Load metrics CSV into a dictionary of numpy arrays.
    Non-numeric columns (like track_type) are stored as-is."""
    import csv
    data: dict[str, list] = {}
    numeric_keys: set[str] = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                    numeric_keys.add(key)
                except (ValueError, TypeError):
                    data[key].append(val)
    # Return numeric columns as numpy arrays, string columns as-is
    result = {}
    for k, v in data.items():
        if k in numeric_keys:
            result[k] = np.array(v, dtype=np.float64)
        else:
            result[k] = v
    return result


def rolling_stats(
    arr: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and std, padding start with NaN."""
    if len(arr) < window:
        mean = np.full_like(arr, np.nan, dtype=np.float64)
        std = np.full_like(arr, np.nan, dtype=np.float64)
        return mean, std

    cumsum = np.cumsum(np.insert(arr, 0, 0))
    mean = (cumsum[window:] - cumsum[:-window]) / window
    mean = np.concatenate([np.full(window - 1, np.nan), mean])

    cumsum2 = np.cumsum(np.insert(arr ** 2, 0, 0))
    var = (cumsum2[window:] - cumsum2[:-window]) / window - mean[window - 1:] ** 2
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    std = np.concatenate([np.full(window - 1, np.nan), std])

    return mean, std


def set_publication_style():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def generate_analytics(
    csv_path: str = "logs/metrics.csv",
    output_dir: str = "figures",
    window: int = 50,
    dpi: int = 150,
) -> str:
    """Generate all analytics plots and summary report.

    Args:
        csv_path: Path to metrics CSV file.
        output_dir: Directory to save figures and report.
        window: Rolling window size for smoothing.
        dpi: DPI for saved figures.

    Returns:
        Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt = set_publication_style()

    if not os.path.exists(csv_path):
        print(f"Error: metrics file not found at {csv_path}")
        return output_dir

    data = load_metrics(csv_path)
    episodes = data["episode"]
    rewards = data["reward"]
    lengths = data["length"]
    crash_rates = data["crash_rate"]
    difficulty = data.get("difficulty", np.zeros_like(rewards))
    center_dist = data.get("center_distance", np.zeros_like(rewards))
    laps = data.get("laps", np.zeros_like(rewards))

    n_episodes = len(episodes)
    print(f"Loaded {n_episodes} episodes from {csv_path}")

    rm, rstd = rolling_stats(rewards, window)
    lm, lstd = rolling_stats(lengths, window)
    cm, _ = rolling_stats(crash_rates, window)

    # ------------------------------------------------------------------
    # Figure 1: Reward curve with rolling mean ± std
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    alpha = 0.08
    ax.scatter(episodes, rewards, s=2, alpha=alpha, color="steelblue", label="Episode reward")
    ax.plot(episodes, rm, color="darkblue", linewidth=1.8, label=f"Rolling mean (w={window})")
    ax.fill_between(
        episodes, rm - rstd, rm + rstd, alpha=0.15, color="darkblue", label=f"±1 std"
    )
    ax.axhline(y=np.mean(rewards), color="crimson", linestyle="--", linewidth=1,
               label=f"Mean: {np.mean(rewards):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Progression")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(os.path.join(output_dir, "reward_progression.png"), dpi=dpi)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Crash rate over time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, cm, color="darkorange", linewidth=1.8, label=f"Rolling crash rate (w={window})")
    ax.fill_between(episodes, np.maximum(cm - 0.05, 0), np.minimum(cm + 0.05, 1),
                    alpha=0.15, color="darkorange")
    ax.axhline(y=np.mean(crash_rates), color="crimson", linestyle="--", linewidth=1,
               label=f"Mean: {np.mean(crash_rates):.3f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Crash Rate")
    ax.set_title("Crash Rate Progression")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(os.path.join(output_dir, "crash_rate.png"), dpi=dpi)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Episode length
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(episodes, lengths, s=2, alpha=0.08, color="mediumseagreen", label="Episode length")
    ax.plot(episodes, lm, color="darkgreen", linewidth=1.8, label=f"Rolling mean (w={window})")
    ax.fill_between(
        episodes, np.maximum(lm - lstd, 0), lm + lstd, alpha=0.15, color="darkgreen",
        label=f"±1 std"
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length Progression")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.savefig(os.path.join(output_dir, "episode_length.png"), dpi=dpi)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Curriculum difficulty
    # ------------------------------------------------------------------
    if np.any(difficulty > 0):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, difficulty, color="mediumpurple", linewidth=1.5,
                label="Difficulty level")
        ax.fill_between(episodes, 0, difficulty, alpha=0.15, color="mediumpurple")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Difficulty")
        ax.set_title("Curriculum Difficulty Progression")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", framealpha=0.9)
        fig.savefig(os.path.join(output_dir, "curriculum_difficulty.png"), dpi=dpi)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: Center distance and laps
    # ------------------------------------------------------------------
    if np.any(center_dist > 0) or np.any(laps > 0):
        fig, ax1 = plt.subplots(figsize=(8, 4))
        cm_dist, _ = rolling_stats(center_dist, window)
        cml, _ = rolling_stats(laps, window)

        ax1.plot(episodes, cm_dist, color="teal", linewidth=1.8, label=f"Center dist (w={window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Center Distance", color="teal")
        ax1.tick_params(axis="y", labelcolor="teal")

        ax2 = ax1.twinx()
        ax2.plot(episodes, cml, color="indianred", linewidth=1.8, label=f"Laps (w={window})")
        ax2.set_ylabel("Laps", color="indianred")
        ax2.tick_params(axis="y", labelcolor="indianred")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)
        ax1.set_title("Center Distance & Laps")
        fig.savefig(os.path.join(output_dir, "center_and_laps.png"), dpi=dpi)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Combined dashboard
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Dashboard", fontsize=15, fontweight="bold")

    # Reward
    ax = axes[0, 0]
    ax.scatter(episodes, rewards, s=2, alpha=0.08, color="steelblue")
    ax.plot(episodes, rm, color="darkblue", linewidth=1.5)
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    # Episode length
    ax = axes[0, 1]
    ax.scatter(episodes, lengths, s=2, alpha=0.08, color="mediumseagreen")
    ax.plot(episodes, lm, color="darkgreen", linewidth=1.5)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")

    # Crash rate
    ax = axes[1, 0]
    ax.plot(episodes, cm, color="darkorange", linewidth=1.5)
    ax.set_title("Crash Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Crash Rate")
    ax.set_ylim(-0.02, 1.02)

    # Difficulty
    ax = axes[1, 1]
    ax.plot(episodes, difficulty, color="mediumpurple", linewidth=1.5)
    ax.set_title("Difficulty")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Difficulty")
    ax.set_ylim(-0.02, max(difficulty.max(), 0.1) * 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_dashboard.png"), dpi=dpi)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Statistical summary report
    # ------------------------------------------------------------------
    report_path = os.path.join(output_dir, "analytics_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Training Analytics Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {csv_path}\n")
        f.write(f"Episodes: {n_episodes}\n")
        f.write(f"Rolling window: {window}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Reward Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:          {rewards.mean():.2f}\n")
        f.write(f"  Std:           {rewards.std():.2f}\n")
        f.write(f"  Min:           {rewards.min():.2f}\n")
        f.write(f"  Max:           {rewards.max():.2f}\n")
        f.write(f"  Median:        {np.median(rewards):.2f}\n")
        f.write(f"  Final (raw):   {rewards[-1]:.2f}\n")
        final_window = rewards[-min(window, n_episodes):]
        f.write(f"  Final {len(final_window)} mean: {final_window.mean():.2f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Episode Length Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:    {lengths.mean():.1f}\n")
        f.write(f"  Max:     {lengths.max():.0f}\n")
        f.write(f"  Min:     {lengths.min():.0f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Crash Rate Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Overall: {crash_rates.mean():.4f} ({crash_rates.mean() * 100:.1f}%)\n")
        first_half = crash_rates[: n_episodes // 2].mean()
        second_half = crash_rates[n_episodes // 2:].mean()
        f.write(f"  First 50%:  {first_half:.4f}\n")
        f.write(f"  Second 50%: {second_half:.4f}\n\n")

        if np.any(laps > 0):
            f.write("-" * 40 + "\n")
            f.write("Lap Statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Mean:    {laps.mean():.2f}\n")
            f.write(f"  Max:     {laps.max():.0f}\n")
            f.write(f"  Final:   {laps[-1]:.0f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Learning Summary\n")
        f.write("-" * 40 + "\n")
        # Improvement from first to last quarter
        q1 = rewards[: n_episodes // 4].mean()
        q4 = rewards[3 * n_episodes // 4:].mean()
        improvement = q4 - q1
        f.write(f"  Q1 mean reward: {q1:.2f}\n")
        f.write(f"  Q4 mean reward: {q4:.2f}\n")
        f.write(f"  Improvement:    {improvement:.2f}\n")
        if crash_rates.mean() > 0:
            f.write(f"  Crash reduction: {(first_half - second_half) * 100:.1f} pp\n")

    print(f"Analytics generated in {output_dir}/")
    print(f"  - reward_progression.png")
    print(f"  - crash_rate.png")
    print(f"  - episode_length.png")
    if np.any(difficulty > 0):
        print(f"  - curriculum_difficulty.png")
    if np.any(center_dist > 0) or np.any(laps > 0):
        print(f"  - center_and_laps.png")
    print(f"  - training_dashboard.png")
    print(f"  - analytics_report.txt")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication-quality training analytics plots"
    )
    parser.add_argument(
        "--csv",
        default="logs/metrics.csv",
        help="Path to metrics CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size for smoothing",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures",
    )
    args = parser.parse_args()
    generate_analytics(args.csv, args.output_dir, args.window, args.dpi)
