#!/usr/bin/env python3
"""
Plot aggregated ablation curves (mean ± std) without overwriting existing figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}_v{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def plot_metric(df, metric_mean, metric_std, title, ylabel, out_path):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(11, 6))
    for variant in sorted(df["variant"].unique()):
        sub = df[df["variant"] == variant]
        x = sub["episode"]
        y = sub[metric_mean]
        yerr = sub[metric_std].fillna(0.0)
        plt.plot(x, y, label=variant)
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path = unique_path(out_path)
    plt.savefig(out_path, format="svg")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="experiments/<timestamp> directory")
    parser.add_argument("--output-dir", default="", help="Optional output directory for figures")
    parser.add_argument("--prefix", default="", help="Optional filename prefix")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    agg_path = in_dir / "aggregate_curves.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}")

    out_dir = Path(args.output_dir) if args.output_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(agg_path)
    prefix = f"{args.prefix}_" if args.prefix else ""

    reward_path = plot_metric(
        df,
        "reward_mean",
        "reward_std",
        "Reward (Mean ± Std) by Variant",
        "Reward",
        out_dir / f"{prefix}reward_mean_std.svg",
    )

    length_path = plot_metric(
        df,
        "length_mean",
        "length_std",
        "Episode Length (Mean ± Std) by Variant",
        "Episode Length",
        out_dir / f"{prefix}episode_length_mean_std.svg",
    )

    crash_free_path = plot_metric(
        df,
        "crash_free_mean",
        "crash_free_std",
        "Crash-Free Rate (Mean ± Std) by Variant",
        "Crash-Free Rate",
        out_dir / f"{prefix}crash_free_rate_mean_std.svg",
    )

    difficulty_path = plot_metric(
        df,
        "difficulty_mean",
        "difficulty_std",
        "Curriculum Difficulty (Mean ± Std) by Variant",
        "Difficulty Level",
        out_dir / f"{prefix}curriculum_difficulty_mean_std.svg",
    )

    print("Figures written:")
    for path in [reward_path, length_path, crash_free_path, difficulty_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
