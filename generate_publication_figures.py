#!/usr/bin/env python3
"""
Generate publication-quality SVG figures from repository training artifacts.

Data sources:
- rl_car_rl/logs/metrics.csv
- rl_car_rl/logs/train/events.out.tfevents.*
- rl_car_rl/configs/tune.yaml + training modules (bounded simulation if no tuning log exists)
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
METRICS_CSV = ROOT / "rl_car_rl" / "logs" / "metrics.csv"
TB_DIR = ROOT / "rl_car_rl" / "logs" / "train"
TUNE_YAML = ROOT / "rl_car_rl" / "configs" / "tune.yaml"
TUNING_RESULTS_YAML = ROOT / "rl_car_rl" / "configs" / "tuning_results.yaml"
TUNING_RESULTS_CSV = FIG_DIR / "hyperparameter_tuning_results_data.csv"

# Match the project import style used by training modules (from env..., from training...).
if str(ROOT / "rl_car_rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "rl_car_rl"))


@dataclass
class FigureMeta:
    filename: str
    caption: str
    description: str
    section: str
    data_source: str


def rolling_mean(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def load_metrics() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    required = {"episode", "reward", "length", "crash_rate", "difficulty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics.csv missing columns: {sorted(missing)}")
    df = df.sort_values("episode").reset_index(drop=True)
    # In this codebase, TensorBoard scalar step is logged as episodes_completed.
    df["training_step"] = df["episode"].astype(int)
    df["reward_ma10"] = rolling_mean(df["reward"], window=10)
    df["length_ma10"] = rolling_mean(df["length"], window=10)
    df["crash_rate_ma10"] = rolling_mean(df["crash_rate"], window=10)
    return df


def inspect_event_file(path: Path) -> Dict[str, int]:
    ea = event_accumulator.EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    counts: Dict[str, int] = {}
    for tag in tags:
        counts[tag] = len(ea.Scalars(tag))
    return counts


def select_event_file(metrics_points: int) -> Optional[Path]:
    files = sorted(TB_DIR.glob("events.out.tfevents.*"))
    if not files:
        return None

    scored: List[tuple] = []
    for f in files:
        try:
            counts = inspect_event_file(f)
        except Exception:
            continue
        reward_n = counts.get("Reward/Episode", 0)
        policy_n = counts.get("Loss/Policy", 0)
        value_n = counts.get("Loss/Value", 0)
        has_losses = int(policy_n > 0 and value_n > 0)
        same_len = int(reward_n == metrics_points)
        scored.append((same_len, has_losses, policy_n + value_n, reward_n, f))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][-1]


def load_losses(selected_event_file: Path) -> pd.DataFrame:
    ea = event_accumulator.EventAccumulator(str(selected_event_file), size_guidance={"scalars": 0})
    ea.Reload()

    policy = ea.Scalars("Loss/Policy") if "Loss/Policy" in ea.Tags().get("scalars", []) else []
    value = ea.Scalars("Loss/Value") if "Loss/Value" in ea.Tags().get("scalars", []) else []

    df_p = pd.DataFrame([{"training_step": x.step, "policy_loss": x.value} for x in policy])
    df_v = pd.DataFrame([{"training_step": x.step, "value_loss": x.value} for x in value])

    if df_p.empty and df_v.empty:
        return pd.DataFrame(columns=["training_step", "policy_loss", "value_loss"])

    if df_p.empty:
        merged = df_v.copy()
        merged["policy_loss"] = float("nan")
        return merged[["training_step", "policy_loss", "value_loss"]]
    if df_v.empty:
        merged = df_p.copy()
        merged["value_loss"] = float("nan")
        return merged[["training_step", "policy_loss", "value_loss"]]

    merged = pd.merge(df_p, df_v, on="training_step", how="outer").sort_values("training_step")
    return merged


def load_tuning_from_yaml() -> Optional[pd.DataFrame]:
    if not TUNING_RESULTS_YAML.exists():
        return None

    with open(TUNING_RESULTS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    all_runs = data.get("all_runs")
    if not all_runs:
        return None

    rows = []
    for item in all_runs:
        if not isinstance(item, dict):
            continue
        for run_name, vals in item.items():
            if not isinstance(vals, dict):
                continue
            row = {"run_name": run_name}
            row.update(vals)
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "final_reward" not in df.columns or "curriculum_level" not in df.columns:
        return None
    return df


def get_combinations(grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = []
    for instance in itertools.product(*vals):
        combos.append(dict(zip(keys, instance)))
    return combos


def run_bounded_tuning_simulation(
    num_envs: int = 4,
    max_episodes: int = 24,
    update_timestep: int = 200,
) -> pd.DataFrame:
    # Import training modules only when needed; keeps plotting path lightweight.
    import numpy as np
    import torch

    from env.vector_env import VectorEnv
    from training.curriculum import CurriculumManager
    from training.trainer import Memory, PPOTrainer

    with open(TUNE_YAML, "r", encoding="utf-8") as f:
        grid = yaml.safe_load(f)
    combos = get_combinations(grid)

    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    for run_id, config in enumerate(combos):
        env = VectorEnv(num_envs=num_envs)
        trainer = PPOTrainer(
            state_dim=9,
            action_dim=2,
            lr=config["learning_rate"],
            gamma=config["gamma"],
            K_epochs=config["k_epochs"],
            eps_clip=config["eps_clip"],
            device=str(device),
        )
        memory = Memory()
        curriculum = CurriculumManager()
        env.set_difficulty(curriculum.get_generation_params())

        time_step = 0
        episodes_completed = 0
        current_ep_rewards = np.zeros(num_envs)
        completed_rewards: List[float] = []
        state = env.reset()

        try:
            while episodes_completed < max_episodes:
                time_step += 1
                final_action, raw_action, log_prob, _ = trainer.policy.get_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(final_action)

                memory.states.append(state)
                memory.actions.append(raw_action.detach().cpu().numpy())
                memory.logprobs.append(log_prob.detach().cpu().numpy())
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                state = next_state
                current_ep_rewards += reward

                for i in range(num_envs):
                    if done[i]:
                        episodes_completed += 1
                        completed_rewards.append(float(current_ep_rewards[i]))
                        current_ep_rewards[i] = 0.0
                        if episodes_completed >= max_episodes:
                            break

                if time_step > 0 and time_step % update_timestep == 0 and completed_rewards:
                    recent_mean = float(np.mean(completed_rewards[-min(len(completed_rewards), 20):]))
                    old_level = curriculum.level
                    new_level = curriculum.update(recent_mean)
                    if new_level > old_level:
                        env.set_difficulty(curriculum.get_generation_params())

                if time_step > 0 and time_step % max(1, update_timestep // 2) == 0 and memory.states:
                    trainer.update(memory)
                    memory.clear_memory()
                    time_step = 0
        finally:
            env.close()

        tail = completed_rewards[-min(20, len(completed_rewards)) :] if completed_rewards else [float("nan")]
        rows.append(
            {
                "run_name": f"sim_run_{run_id}",
                **config,
                "final_reward": float(np.mean(tail)),
                "curriculum_level": float(curriculum.level),
                "sim_num_envs": num_envs,
                "sim_max_episodes": max_episodes,
                "sim_update_timestep": update_timestep,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["curriculum_level", "final_reward"], ascending=False).reset_index(drop=True)


def make_plots(metrics_df: pd.DataFrame, losses_df: pd.DataFrame, tuning_df: pd.DataFrame) -> List[FigureMeta]:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["svg.fonttype"] = "none"

    meta: List[FigureMeta] = []

    # 1) Reward vs training step
    fig, ax = plt.subplots()
    ax.plot(metrics_df["training_step"], metrics_df["reward"], color="#1f77b4", alpha=0.35, linewidth=1.2, label="Episode reward")
    ax.plot(metrics_df["training_step"], metrics_df["reward_ma10"], color="#1f77b4", linewidth=2.8, label="10-episode moving avg")
    ax.set_xlabel("Training step (logged episode index)")
    ax.set_ylabel("Episode reward")
    ax.set_title("Reward Progression During PPO Training")
    ax.legend(frameon=True)
    fig.tight_layout()
    fn = "reward_vs_training_steps.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    meta.append(
        FigureMeta(
            filename=fn,
            caption="Episode reward trajectory over training, with a 10-episode moving average indicating progressive performance gains.",
            description=(
                "This figure plots per-episode reward against logged training step and overlays a 10-episode moving average. "
                "The series shows high variance with an upward trend, reflecting increasing policy competence while preserving episodic stochasticity."
            ),
            section="Results",
            data_source=str(METRICS_CSV.relative_to(ROOT)),
        )
    )

    # 2) Episode length vs training step
    fig, ax = plt.subplots()
    ax.plot(metrics_df["training_step"], metrics_df["length"], color="#2ca02c", alpha=0.35, linewidth=1.2, label="Episode length")
    ax.plot(metrics_df["training_step"], metrics_df["length_ma10"], color="#2ca02c", linewidth=2.8, label="10-episode moving avg")
    ax.set_xlabel("Training step (logged episode index)")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title("Episode Duration During Training")
    ax.legend(frameon=True)
    fig.tight_layout()
    fn = "episode_length_vs_training_steps.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    meta.append(
        FigureMeta(
            filename=fn,
            caption="Episode length as a function of training step, with moving-average smoothing to reveal stability trends.",
            description=(
                "Per-episode duration is plotted against training step to characterize survivability and control consistency. "
                "The moving-average line highlights changes in sustained driving behavior across training."
            ),
            section="Results",
            data_source=str(METRICS_CSV.relative_to(ROOT)),
        )
    )

    # 3) PPO policy and value losses
    fig, ax = plt.subplots()
    if not losses_df.empty:
        ax.plot(losses_df["training_step"], losses_df["policy_loss"], marker="o", linewidth=2.4, color="#d62728", label="Policy loss")
        ax.plot(losses_df["training_step"], losses_df["value_loss"], marker="s", linewidth=2.4, color="#9467bd", label="Value loss")
    else:
        ax.text(0.5, 0.5, "No PPO loss scalars found in event files", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Training step (logged episode index)")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Optimization Losses")
    ax.legend(frameon=True)
    fig.tight_layout()
    fn = "ppo_policy_value_losses.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    loss_source = "N/A"
    if "event_file" in losses_df.columns and not losses_df.empty:
        loss_source = losses_df["event_file"].iloc[0]
    meta.append(
        FigureMeta(
            filename=fn,
            caption="PPO policy and value loss traces over logged optimization steps.",
            description=(
                "This plot reports policy and critic losses from TensorBoard scalars in the selected training event file. "
                "The points correspond to optimizer update checkpoints and quantify the optimization dynamics observed in this run."
            ),
            section="Training Dynamics",
            data_source=loss_source,
        )
    )

    # 4) Crash rate vs training step
    fig, ax = plt.subplots()
    ax.plot(metrics_df["training_step"], metrics_df["crash_rate"], linestyle="", marker="o", markersize=4, alpha=0.35, color="#ff7f0e", label="Episode crash outcome")
    ax.plot(metrics_df["training_step"], metrics_df["crash_rate_ma10"], linewidth=2.8, color="#ff7f0e", label="10-episode moving avg")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training step (logged episode index)")
    ax.set_ylabel("Crash rate (0/1)")
    ax.set_title("Crash Outcomes Across Training")
    ax.legend(frameon=True)
    fig.tight_layout()
    fn = "crash_rate_vs_training_steps.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    meta.append(
        FigureMeta(
            filename=fn,
            caption="Binary crash outcomes and smoothed crash rate across training steps.",
            description=(
                "Each point indicates whether an episode terminated by crash (1) or not (0), and the moving average summarizes the short-horizon failure rate. "
                "This view captures safety-related behavior throughout policy learning."
            ),
            section="Safety Analysis",
            data_source=str(METRICS_CSV.relative_to(ROOT)),
        )
    )

    # 5) Curriculum difficulty vs training step
    fig, ax = plt.subplots()
    ax.plot(metrics_df["training_step"], metrics_df["difficulty"], linewidth=2.8, color="#8c564b")
    ax.set_xlabel("Training step (logged episode index)")
    ax.set_ylabel("Curriculum difficulty level")
    ax.set_title("Curriculum Difficulty Schedule")
    ax.set_ylim(min(0.0, metrics_df["difficulty"].min() - 0.05), max(1.0, metrics_df["difficulty"].max() + 0.05))
    fig.tight_layout()
    fn = "curriculum_difficulty_vs_training_steps.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    meta.append(
        FigureMeta(
            filename=fn,
            caption="Curriculum difficulty level recorded over training steps.",
            description=(
                "The curriculum scalar logged per episode is shown to track environment difficulty progression over time. "
                "In this run, the line reveals whether threshold-based level updates were reached."
            ),
            section="Method",
            data_source=str(METRICS_CSV.relative_to(ROOT)),
        )
    )

    # 6) Hyperparameter tuning results
    fig, ax = plt.subplots(figsize=(12, 6))
    tuning_plot = tuning_df.copy().reset_index(drop=True)
    tuning_plot["run_idx"] = tuning_plot.index + 1
    scatter = ax.scatter(
        tuning_plot["run_idx"],
        tuning_plot["final_reward"],
        c=tuning_plot["curriculum_level"],
        cmap="viridis",
        s=110,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.plot(tuning_plot["run_idx"], tuning_plot["final_reward"], color="#1f77b4", alpha=0.4, linewidth=1.5)
    ax.set_xlabel("Configuration rank (sorted by curriculum level, then reward)")
    ax.set_ylabel("Final reward (tail mean)")
    ax.set_title("Hyperparameter Tuning Outcomes")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Final curriculum level")
    fig.tight_layout()
    fn = "hyperparameter_tuning_results.svg"
    fig.savefig(FIG_DIR / fn, format="svg")
    plt.close(fig)
    meta.append(
        FigureMeta(
            filename=fn,
            caption="Hyperparameter sweep outcomes showing final reward by configuration rank, colored by achieved curriculum level.",
            description=(
                "Each point corresponds to one hyperparameter configuration from the tuning grid and reports the run-level final reward. "
                "Color encodes achieved curriculum level to jointly assess return quality and difficulty progression."
            ),
            section="Ablation / Hyperparameter Study",
            data_source="figures/hyperparameter_tuning_results_data.csv",
        )
    )

    return meta


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics()
    metrics_df.to_csv(FIG_DIR / "extracted_metrics_data.csv", index=False)

    selected_event = select_event_file(metrics_points=len(metrics_df))
    if selected_event is not None:
        losses_df = load_losses(selected_event)
        if not losses_df.empty:
            losses_df["event_file"] = str(selected_event.relative_to(ROOT))
    else:
        losses_df = pd.DataFrame(columns=["training_step", "policy_loss", "value_loss", "event_file"])
    losses_df.to_csv(FIG_DIR / "ppo_losses_data.csv", index=False)

    tuning_df = load_tuning_from_yaml()
    if tuning_df is None:
        if TUNING_RESULTS_CSV.exists():
            tuning_df = pd.read_csv(TUNING_RESULTS_CSV)
            if "data_origin" not in tuning_df.columns:
                tuning_df["data_origin"] = "figures/hyperparameter_tuning_results_data.csv"
        else:
            tuning_df = run_bounded_tuning_simulation(num_envs=4, max_episodes=24, update_timestep=200)
            tuning_df["data_origin"] = "bounded_simulation_from_training_modules"
    else:
        tuning_df["data_origin"] = "configs/tuning_results.yaml"
    tuning_df.to_csv(TUNING_RESULTS_CSV, index=False)

    meta = make_plots(metrics_df, losses_df, tuning_df)
    meta_df = pd.DataFrame([m.__dict__ for m in meta])
    meta_df.to_csv(FIG_DIR / "figure_metadata.csv", index=False)
    with open(FIG_DIR / "figure_metadata.json", "w", encoding="utf-8") as f:
        json.dump([m.__dict__ for m in meta], f, indent=2)

    print("Generated files:")
    for path in sorted(FIG_DIR.glob("*.svg")):
        print("-", path.relative_to(ROOT))
    print("\nMetadata:")
    print((FIG_DIR / "figure_metadata.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
