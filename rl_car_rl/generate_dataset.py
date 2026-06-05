"""
Dataset generation: records state-action-reward trajectories for offline RL.
Outputs compressed NPZ files with (states, actions, rewards, next_states, dones).
"""

import argparse
import os
import sys
import numpy as np
import yaml
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import CarEnv
from training.trainer import PPOTrainer


def generate_dataset(
    num_episodes: int = 100,
    output_path: str = "datasets/trajectories.npz",
    checkpoint: str | None = None,
    deterministic: bool = False,
) -> str:
    """
    Generate a dataset of state-action-reward trajectories.

    Args:
        num_episodes: Number of episodes to collect.
        output_path: Path to save the .npz file.
        checkpoint: Path to a trained checkpoint (random policy if None).
        deterministic: Use deterministic actions.

    Returns:
        Path to the saved dataset.
    """
    config_path = os.path.join("configs", "hyperparameters.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    sensor_count = config.get("sensor_count", 16)
    obstacle_count = config.get("obstacle_count", 0)
    state_dim = sensor_count + 4

    env = CarEnv(sensor_count=sensor_count, obstacle_count=obstacle_count)

    trainer = None
    if checkpoint:
        trainer = PPOTrainer(state_dim=state_dim, action_dim=2)
        if not trainer.load(checkpoint):
            print("Checkpoint not found, using random policy.")
            trainer = None

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    episode_rewards = []
    total_steps = 0

    print(f"Generating {num_episodes} episodes of trajectory data...")

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            if trainer is not None:
                action, _, _, _ = trainer.policy.get_action(
                    state, deterministic=deterministic
                )
            else:
                action = np.array([
                    np.random.uniform(-1, 1),  # steering
                    np.random.uniform(0, 1),    # throttle
                ], dtype=np.float32)

            next_state, reward, done, _ = env.step(action)

            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_state)
            all_dones.append(done)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

        episode_rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep + 1}/{num_episodes} | "
                  f"Reward: {ep_reward:.1f} | Steps: {ep_steps} | "
                  f"Total: {total_steps}")

    # Save as compressed NPZ
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_states=np.array(all_next_states, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
    )

    print(f"\nDataset saved to {output_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} +- {np.std(episode_rewards):.2f}")
    print(f"  States dim: {state_dim}, Actions dim: 2")
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.1f} KB" if file_size < 1e6
          else f"  File size: {file_size / 1e6:.1f} MB")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectory dataset for offline RL"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="datasets/trajectories.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    generate_dataset(
        args.episodes, args.output, args.checkpoint, args.deterministic
    )
