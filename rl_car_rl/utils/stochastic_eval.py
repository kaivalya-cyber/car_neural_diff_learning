import numpy as np
import os
import yaml


def stochastic_evaluate(env, trainer, num_episodes: int = 20, num_seeds: int = 5) -> dict:
    all_rewards = []
    all_steps = []
    all_crashes = []
    all_laps = []

    for seed in range(num_seeds):
        np.random.seed(seed)
        ep_rewards = []
        ep_steps = []
        ep_crashes = []
        ep_laps = []

        for ep in range(num_episodes):
            state = env.reset()
            trainer.policy.reset_noise()
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

            ep_rewards.append(total_reward)
            ep_steps.append(steps)
            ep_crashes.append(info.get("crashed", False))
            ep_laps.append(info.get("lap_count", 0))

        all_rewards.append(ep_rewards)
        all_steps.append(ep_steps)
        all_crashes.append(ep_crashes)
        all_laps.append(ep_laps)

    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    all_crashes = np.array(all_crashes)
    all_laps = np.array(all_laps)

    return {
        "reward_mean": float(all_rewards.mean()),
        "reward_std": float(all_rewards.std()),
        "reward_across_seeds_mean": float(all_rewards.mean(axis=1).mean()),
        "reward_across_seeds_std": float(all_rewards.mean(axis=1).std()),
        "steps_mean": float(all_steps.mean()),
        "crash_rate": float(all_crashes.mean()),
        "laps_mean": float(all_laps.mean()),
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
    }


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Stochastic evaluation across seeds")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.environment import CarEnv
    from training.trainer import PPOTrainer

    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sensor_count = config.get("sensor_count", 16)
    state_dim = sensor_count + 4
    env = CarEnv(sensor_count=sensor_count)
    trainer = PPOTrainer(state_dim=state_dim, action_dim=2)
    loaded = trainer.load(args.checkpoint)
    if not loaded:
        print("Warning: running with random weights")

    results = stochastic_evaluate(env, trainer, args.episodes, args.seeds)
    print("=" * 50)
    print("Stochastic Evaluation Results")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
