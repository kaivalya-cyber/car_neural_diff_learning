import csv
import os
import numpy as np
from datetime import datetime


class TelemetryRecorder:
    def __init__(self, output_dir: str = "telemetry"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fields = [
            "timestamp", "episode", "step", "x", "y", "heading",
            "velocity", "steering", "throttle", "reward", "lap_count",
            "progress", "center_distance", "crashed", "sensor_0",
            "sensor_1", "sensor_2", "sensor_3", "sensor_4",
        ]
        self.file = None
        self.writer = None
        self.current_episode = -1

    def start_episode(self, episode: int):
        self.current_episode = episode
        if self.file:
            self.file.close()
        path = os.path.join(self.output_dir, f"episode_{episode:06d}.csv")
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def record_step(self, episode: int, step: int, car, reward: float, done: bool, info: dict,
                    sensor_readings: np.ndarray | None = None):
        if episode != self.current_episode:
            self.start_episode(episode)

        row = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "step": step,
            "x": round(car.x, 2),
            "y": round(car.y, 2),
            "heading": round(car.heading, 4),
            "velocity": round(car.velocity, 2),
            "steering": round(car.steering_angle, 4),
            "throttle": round(car.throttle, 4),
            "reward": round(reward, 4),
            "lap_count": info.get("lap_count", 0),
            "progress": round(info.get("progress", 0), 4),
            "center_distance": round(info.get("center_distance", 0), 4),
            "crashed": info.get("crashed", False),
        }
        if sensor_readings is not None:
            for i in range(min(5, len(sensor_readings))):
                row[f"sensor_{i}"] = round(float(sensor_readings[i]), 4)

        if self.writer:
            self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def main():
    import argparse, sys, yaml, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output", default="telemetry")
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
    trainer.load(args.checkpoint)

    recorder = TelemetryRecorder(args.output)
    for ep in range(args.episodes):
        state = env.reset()
        recorder.start_episode(ep)
        step = 0
        total_reward = 0
        while True:
            action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            recorder.record_step(ep, step, env.car, reward, done, info, state[:sensor_count])
            step += 1
            if done:
                print(f"  Ep {ep}: {step} steps, reward={total_reward:.1f}")
                break
    recorder.close()
    print(f"Telemetry saved to {args.output}/")


if __name__ == "__main__":
    main()
