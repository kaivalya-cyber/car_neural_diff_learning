#!/usr/bin/env python3
"""
Run ablation studies without overwriting existing logs/checkpoints.

Outputs are written to experiments/<timestamp>/...
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project imports resolve regardless of cwd
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "rl_car_rl"))

from training.training_loop import train_agent

THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1.0]


def threshold_key(threshold: float) -> str:
    return f"threshold_{int(threshold * 100):02d}_episode"


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_experiments(output_root: Path, seeds: list[int], max_episodes: int, num_envs_vec: int, only_variants=None):
    variants = [
        {"name": "curriculum_vectorized", "use_curriculum": True, "num_envs": num_envs_vec},
        {"name": "no_curriculum_vectorized", "use_curriculum": False, "num_envs": num_envs_vec},
        {"name": "curriculum_single_env", "use_curriculum": True, "num_envs": 1},
        {"name": "no_curriculum_single_env", "use_curriculum": False, "num_envs": 1},
    ]
    if only_variants:
        variants = [v for v in variants if v["name"] in only_variants]

    runs = []
    for variant in variants:
        for seed in seeds:
            run_dir = output_root / variant["name"] / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "logs" / "metrics.csv"
            if metrics_path.exists():
                print(f"Skipping existing run: {variant['name']} seed={seed}")
                runs.append(
                    {
                        "variant": variant["name"],
                        "seed": seed,
                        "num_envs": variant["num_envs"],
                        "use_curriculum": variant["use_curriculum"],
                        "run_dir": str(run_dir),
                    }
                )
                continue

            config_overrides = {
                "num_envs": variant["num_envs"],
                "max_episodes": max_episodes,
            }

            print(f"\n=== Running {variant['name']} seed={seed} ===")
            train_agent(
                config_overrides=config_overrides,
                output_dir=str(run_dir),
                use_curriculum=variant["use_curriculum"],
                seed=seed,
            )

            runs.append(
                {
                    "variant": variant["name"],
                    "seed": seed,
                    "num_envs": variant["num_envs"],
                    "use_curriculum": variant["use_curriculum"],
                    "run_dir": str(run_dir),
                }
            )

    return runs


def aggregate_runs(output_root: Path, runs: list[dict]):
    rows = []
    curves = []
    threshold_rows = []

    for run in runs:
        metrics_path = Path(run["run_dir"]) / "logs" / "metrics.csv"
        if not metrics_path.exists():
            print(f"Missing metrics: {metrics_path}")
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue

        df["crash_free"] = 1.0 - df["crash_rate"]
        df["variant"] = run["variant"]
        df["seed"] = run["seed"]
        curves.append(df)

        # Curriculum progression: first episode where difficulty > 0
        advanced = df[df["difficulty"] > 0.0]
        first_advance = int(advanced["episode"].iloc[0]) if not advanced.empty else None
        threshold_hits = {}
        for threshold in THRESHOLDS:
            hit = df[df["difficulty"] >= threshold]
            threshold_hits[threshold_key(threshold)] = int(hit["episode"].iloc[0]) if not hit.empty else None

        rows.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                "num_envs": run["num_envs"],
                "use_curriculum": run["use_curriculum"],
                "episodes": int(df["episode"].max()),
                "final_reward_mean_10": float(df["reward"].rolling(10, min_periods=1).mean().iloc[-1]),
                "final_length_mean_10": float(df["length"].rolling(10, min_periods=1).mean().iloc[-1]),
                "final_crash_free_mean_10": float(df["crash_free"].rolling(10, min_periods=1).mean().iloc[-1]),
                "curriculum_first_advance_episode": first_advance,
                "final_curriculum_level": float(df["difficulty"].iloc[-1]),
                "metrics_path": str(metrics_path),
                **threshold_hits,
            }
        )
        threshold_rows.append(
            {
                "variant": run["variant"],
                "seed": run["seed"],
                **threshold_hits,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "summary.csv", index=False)

    if threshold_rows:
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_df.to_csv(output_root / "curriculum_thresholds.csv", index=False)

        summary_rows = []
        for variant in threshold_df["variant"].unique():
            sub = threshold_df[threshold_df["variant"] == variant]
            row = {"variant": variant}
            for threshold in THRESHOLDS:
                key = threshold_key(threshold)
                row[f"{key}_mean"] = float(sub[key].mean()) if key in sub else None
                row[f"{key}_std"] = float(sub[key].std()) if key in sub else None
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(output_root / "curriculum_thresholds_summary.csv", index=False)

    if not curves:
        return

    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(output_root / "all_curves.csv", index=False)

    # Aggregate mean/std per episode per variant
    agg = (
        all_curves.groupby(["variant", "episode"])
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            length_mean=("length", "mean"),
            length_std=("length", "std"),
            crash_free_mean=("crash_free", "mean"),
            crash_free_std=("crash_free", "std"),
            difficulty_mean=("difficulty", "mean"),
            difficulty_std=("difficulty", "std"),
        )
        .reset_index()
    )
    agg.to_csv(output_root / "aggregate_curves.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="", help="Reuse an existing experiments directory")
    parser.add_argument("--variants", type=str, default="", help="Comma-separated variant names to run")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / "experiments" / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    only_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    runs = run_experiments(out_dir, seeds, args.max_episodes, args.num_envs, only_variants or None)
    with open(out_dir / "runs.json", "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    aggregate_runs(out_dir, runs)
    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
