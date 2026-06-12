"""
Automated ablation study runner.

Runs short training sessions across hyperparameter sweep dimensions,
generates comparison analytics with summary table and charts.

Usage:
    python ablate.py --param learning_rate --values 0.0001,0.0003,0.001
    python ablate.py --param num_blocks --values 1,2,3,4
    python ablate.py --episodes 200 --output-dir ablate_results/
"""

import os
import sys
import yaml
import argparse
import time
import copy
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_ablation_study(
    param: str,
    values: list,
    episodes: int = 200,
    output_dir: str = "ablate_results",
    base_config_overrides: dict = None,
):
    """Run parameter sweep and generate comparison report."""
    from training.training_loop import train_agent

    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"Ablation study: {param} = {values}")
    print(f"  Episodes per run: {episodes}")
    print(f"  Output directory: {output_dir}/")
    print()

    for val in values:
        run_name = f"{param}_{val}"
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        overrides = {"max_episodes": episodes, param: val,
                     "early_stop_patience": episodes + 10}
        if base_config_overrides:
            overrides.update(base_config_overrides)

        print(f"--- Running {param}={val} ---")
        try:
            train_agent(config_overrides=overrides, output_dir=run_dir,
                       use_curriculum=False)
            result = _extract_result(run_dir)
            result["param"] = param
            result["value"] = val
            results.append(result)
            print(f"  Best reward: {result['best_reward']:.1f}, "
                  f"Episodes: {result['episodes_completed']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"param": param, "value": val,
                           "best_reward": float("-inf"),
                           "episodes_completed": 0, "error": str(e)})

    # Generate report
    _generate_report(results, param, output_dir)
    return results


def _extract_result(run_dir: str) -> dict:
    """Extract best reward and episode count from a training run's CSV."""
    csv_path = os.path.join(run_dir, "logs", "metrics.csv")
    if not os.path.exists(csv_path):
        return {"best_reward": float("-inf"), "episodes_completed": 0}

    rewards = []
    with open(csv_path, "r") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row.get("reward", 0)))

    return {
        "best_reward": max(rewards) if rewards else float("-inf"),
        "episodes_completed": len(rewards),
        "final_reward": rewards[-1] if rewards else 0,
        "mean_reward": np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
    }


def _generate_report(results: list, param: str, output_dir: str):
    """Generate a summary table and bar chart."""
    valid = [r for r in results if r["best_reward"] > float("-inf")]
    if not valid:
        print("No valid results to report.")
        return

    # Sort by best reward
    valid.sort(key=lambda r: r["best_reward"], reverse=True)
    best = valid[0]

    report_path = os.path.join(output_dir, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Ablation Study: {param}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tested values: {[r['value'] for r in valid]}\n\n")
        f.write(f"{'Value':>12}  {'Best Reward':>12}  {'Mean (last50)':>14}  {'Episodes':>10}\n")
        f.write("-" * 52 + "\n")
        for r in valid:
            f.write(f"{str(r['value']):>12}  {r['best_reward']:>12.1f}  "
                    f"{r.get('mean_reward', 0):>14.1f}  {r['episodes_completed']:>10}\n")
        f.write("\n")
        f.write(f"Best: {param}={best['value']} (reward={best['best_reward']:.1f})\n")

    print(f"\nAblation report saved to {report_path}")

    # Generate bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values_str = [str(r["value"]) for r in valid]
        rewards_val = [r["best_reward"] for r in valid]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(values_str, rewards_val, color="steelblue", edgecolor="darkblue")
        ax.set_xlabel(param)
        ax.set_ylabel("Best Reward")
        ax.set_title(f"Ablation Study: {param}")
        ax.grid(axis="y", alpha=0.3)

        # Highlight best
        if bars:
            bars[0].set_color("darkorange")

        chart_path = os.path.join(output_dir, "ablation_chart.png")
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved to {chart_path}")
    except Exception as e:
        print(f"Chart generation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter ablation study runner")
    parser.add_argument("--param", required=True, help="Parameter name to sweep")
    parser.add_argument("--values", required=True, help="Comma-separated values (e.g. 1,2,3 or 0.001,0.01,0.1)")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per run")
    parser.add_argument("--output-dir", default="ablate_results", help="Output directory")
    args = parser.parse_args()

    # Parse values, preserving type
    def parse_val(v):
        v = v.strip()
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    values = [parse_val(v) for v in args.values.split(",")]
    run_ablation_study(args.param, values, args.episodes, args.output_dir)
