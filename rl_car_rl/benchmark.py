"""
Automated benchmarking: evaluate multiple checkpoints and generate comparison reports.
"""

import argparse
import os
import sys
import json
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import CarEnv
from training.trainer import PPOTrainer


def evaluate_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    sensor_count: int,
    obstacle_count: int,
    num_episodes: int = 10,
) -> dict:
    """Evaluate a single checkpoint and return metrics."""
    env = CarEnv(sensor_count=sensor_count, obstacle_count=obstacle_count)
    trainer = PPOTrainer(state_dim=state_dim, action_dim=2)
    loaded = trainer.load(checkpoint_path)

    rewards = []
    steps_list = []
    crashes = []
    laps_list = []

    for _ in range(num_episodes):
        state = env.reset()
        trainer.policy.reset_noise()
        ep_reward, ep_steps = 0.0, 0

        while True:
            action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if done:
                rewards.append(ep_reward)
                steps_list.append(ep_steps)
                crashes.append(info.get("crashed", False))
                laps_list.append(info.get("lap_count", 0))
                break

    rewards_arr = np.array(rewards)
    return {
        "checkpoint": checkpoint_path,
        "loaded": loaded,
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "median_reward": float(np.median(rewards_arr)),
        "max_reward": float(rewards_arr.max()),
        "mean_steps": float(np.mean(steps_list)),
        "mean_laps": float(np.mean(laps_list)),
        "crash_rate": float(np.mean(crashes)),
        "num_episodes": num_episodes,
    }


def run_benchmark(
    checkpoints: list[str],
    state_dim: int,
    sensor_count: int = 16,
    obstacle_count: int = 0,
    num_episodes: int = 10,
    output_path: str = "",
) -> list[dict]:
    """Run benchmark across multiple checkpoints."""
    results = []
    print(f"Benchmarking {len(checkpoints)} checkpoints "
          f"({num_episodes} episodes each)...\n")

    for i, ckpt in enumerate(checkpoints):
        if not os.path.exists(ckpt):
            print(f"  [{i + 1}/{len(checkpoints)}] {ckpt} - NOT FOUND, skipping")
            results.append({"checkpoint": ckpt, "loaded": False})
            continue

        print(f"  [{i + 1}/{len(checkpoints)}] {ckpt}")
        result = evaluate_checkpoint(
            ckpt, state_dim, sensor_count, obstacle_count, num_episodes
        )
        results.append(result)
        print(f"    Reward: {result['mean_reward']:.2f} +- {result['std_reward']:.2f} "
              f"| Laps: {result['mean_laps']:.1f} | Crash: {result['crash_rate']:.1%}")

    # Sort by mean reward descending
    results.sort(key=lambda r: r.get("mean_reward", -float("inf")), reverse=True)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Rank':<5} {'Checkpoint':<30} {'Reward':>10} {'Laps':>8} {'Crash':>8}")
    print("=" * 70)
    for rank, r in enumerate(results):
        if r.get("loaded"):
            print(
                f"{rank + 1:<5} {os.path.basename(r['checkpoint']):<30} "
                f"{r['mean_reward']:>8.1f}+-{r['std_reward']:>5.1f} "
                f"{r['mean_laps']:>6.1f} {r['crash_rate']:>7.1%}"
            )
        else:
            print(f"{rank + 1:<5} {os.path.basename(r['checkpoint']):<30} {'N/A':>10}")
    print("=" * 70)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def discover_checkpoints(checkpoint_dir: str = "checkpoints") -> list[str]:
    """Find all checkpoint files in a directory."""
    if not os.path.exists(checkpoint_dir):
        return []
    files = []
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith(".pth"):
            files.append(os.path.join(checkpoint_dir, f))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trained checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Specific checkpoint files to benchmark")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory to scan for .pth files")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sensor-count", type=int, default=16)
    parser.add_argument("--obstacle-count", type=int, default=0)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--output", default="", help="Save results as JSON")
    args = parser.parse_args()

    checkpoints = args.checkpoints or discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)

    run_benchmark(
        checkpoints=checkpoints,
        state_dim=args.state_dim,
        sensor_count=args.sensor_count,
        obstacle_count=args.obstacle_count,
        num_episodes=args.episodes,
        output_path=args.output,
    )
