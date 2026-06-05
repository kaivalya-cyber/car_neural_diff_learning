import argparse
import sys
import yaml
import os
import time
import numpy as np

# Ensure packages can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.training_loop import train_agent
from training.tuner import tune_hyperparameters
from training.trainer import PPOTrainer
from env.environment import CarEnv
from visualization.renderer import Renderer


def evaluate(num_episodes: int = 10, render: bool = True) -> None:
    """Run evaluation episodes and report aggregate statistics."""
    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sensor_count = config.get("sensor_count", 16)
    obstacle_count = config.get("obstacle_count", 0)
    state_dim = sensor_count + 4

    env = CarEnv(sensor_count=sensor_count, obstacle_count=obstacle_count)
    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=2,
        hidden_size=config.get("hidden_size", 256),
        num_blocks=config.get("num_blocks", 2),
        dropout=config.get("dropout", 0.0),
    )

    if not trainer.load("checkpoints/latest.pth"):
        print("Running with random weights...")

    renderer = Renderer(env, fps=60) if render else None

    rewards = []
    steps_list = []
    crashes = []
    laps_list = []

    print(f"Running {num_episodes} evaluation episodes...")

    for ep in range(num_episodes):
        state = env.reset()
        trainer.policy.reset_noise()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            final_action, _, _, _ = trainer.policy.get_action(
                state, deterministic=True
            )
            state, reward, done, info = env.step(final_action)
            ep_reward += reward
            ep_steps += 1

            if render:
                if not renderer.render(reward=reward, done=done):
                    if renderer:
                        renderer.close()
                    return

            if done:
                crashed = info.get("crashed", False)
                laps = info.get("lap_count", 0)
                rewards.append(ep_reward)
                steps_list.append(ep_steps)
                crashes.append(crashed)
                laps_list.append(laps)
                print(
                    f"  Ep {ep + 1}/{num_episodes}: "
                    f"Reward={ep_reward:.1f} Steps={ep_steps} "
                    f"Crashed={'yes' if crashed else 'no'} Laps={laps}"
                )
                time.sleep(0.5)
                break

    if renderer:
        renderer.close()

    # Statistics
    rewards = np.array(rewards)
    steps_list = np.array(steps_list, dtype=float)
    laps_list = np.array(laps_list, dtype=float)
    crash_rate = np.mean(crashes)

    print("\n" + "=" * 50)
    print(f"Evaluation Results ({num_episodes} episodes)")
    print("=" * 50)
    print(f"  Reward:    {rewards.mean():.2f} ± {rewards.std():.2f}  [min={rewards.min():.2f}, max={rewards.max():.2f}, median={np.median(rewards):.2f}]")
    print(f"  Steps:     {steps_list.mean():.1f} ± {steps_list.std():.1f}")
    print(f"  Laps:      {laps_list.mean():.1f} ± {laps_list.std():.1f}")
    print(f"  Crash Rate: {crash_rate * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Car Agent Execution")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "tune"],
        default="train",
        help="Mode to run: train, evaluate, or tune",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (evaluate mode only)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during evaluation",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=16,
        help="Number of tuning trials (tune mode only)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_agent(resume=args.resume)
    elif args.mode == "evaluate":
        evaluate(num_episodes=args.episodes, render=not args.no_render)
    elif args.mode == "tune":
        tune_hyperparameters(budget=args.budget)
