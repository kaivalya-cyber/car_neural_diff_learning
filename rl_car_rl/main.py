import argparse
import sys
import yaml
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.training_loop import train_agent
from training.tuner import tune_hyperparameters
from training.trainer import PPOTrainer
from training.experiment_tracker import ExperimentTracker
from env.environment import CarEnv
from env.multi_car_env import MultiCarEnv
from visualization.renderer import Renderer
from visualization.multi_renderer import MultiCarRenderer


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

            if render and renderer:
                if not renderer.render(reward=reward, done=done):
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

    rewards = np.array(rewards)
    steps_list = np.array(steps_list, dtype=float)
    laps_list = np.array(laps_list, dtype=float)
    crash_rate = np.mean(crashes)

    print("\n" + "=" * 50)
    print(f"Evaluation Results ({num_episodes} episodes)")
    print("=" * 50)
    print(f"  Reward:    {rewards.mean():.2f} +- {rewards.std():.2f}  [min={rewards.min():.2f}, max={rewards.max():.2f}, median={np.median(rewards):.2f}]")
    print(f"  Steps:     {steps_list.mean():.1f} +- {steps_list.std():.1f}")
    print(f"  Laps:      {laps_list.mean():.1f} +- {laps_list.std():.1f}")
    print(f"  Crash Rate: {crash_rate * 100:.1f}%")
    print("=" * 50)


def race(render: bool = True) -> None:
    """Run multi-car racing mode with 2 cars on the same track."""
    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sensor_count = config.get("sensor_count", 16)
    obstacle_count = config.get("obstacle_count", 0)
    state_dim = sensor_count + 4

    env = MultiCarEnv(
        num_cars=2,
        sensor_count=sensor_count,
        obstacle_count=obstacle_count,
    )

    # Load two policies (or share one)
    trainer1 = PPOTrainer(
        state_dim=state_dim, action_dim=2,
        hidden_size=config.get("hidden_size", 256),
    )
    trainer2 = PPOTrainer(
        state_dim=state_dim, action_dim=2,
        hidden_size=config.get("hidden_size", 256),
    )

    loaded1 = trainer1.load("checkpoints/latest.pth")
    loaded2 = trainer2.load("checkpoints/latest.pth")

    if not loaded1:
        print("Car 1: running with random weights...")
    if not loaded2:
        print("Car 2: running with random weights...")

    renderer = MultiCarRenderer(env, fps=60) if render else None
    obs = env.reset()

    running = True
    print("Starting multi-car race. Close window to exit.")

    while running:
        action1, _, _, _ = trainer1.policy.get_action(obs[0], deterministic=True)
        action2, _, _, _ = trainer2.policy.get_action(obs[1], deterministic=True)
        actions = np.stack([action1, action2])

        obs, rewards, dones, infos = env.step(actions)

        if render and renderer:
            running = renderer.render(done=any(dones))

        if any(dones):
            for i, done in enumerate(dones):
                if done:
                    print(
                        f"  {env.cars[i].name}: crashed={infos[i].get('crashed', False)}, "
                        f"laps={infos[i].get('lap_count', 0)}, "
                        f"reward={rewards[i]:.1f}"
                    )
            obs = env.reset()
            time.sleep(1)

    if renderer:
        renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Car Agent Execution")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "tune", "race", "export"],
        default="train",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output", default="exported/model.pt")
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    if args.list_experiments:
        from training.experiment_tracker import list_experiments
        exps = list_experiments()
        print(f"Found {len(exps)} experiments:")
        for exp in exps:
            print(f"  {exp['id']}: best={exp['best_reward']}, "
                  f"eps={exp['episodes']}, status={exp['status']}")
    elif args.mode == "train":
        train_agent(resume=args.resume)
    elif args.mode == "evaluate":
        evaluate(num_episodes=args.episodes, render=not args.no_render)
    elif args.mode == "tune":
        tune_hyperparameters(budget=args.budget)
    elif args.mode == "race":
        race(render=not args.no_render)
    elif args.mode == "export":
        from export import export_model
        export_model(args.checkpoint, args.output, args.state_dim)
