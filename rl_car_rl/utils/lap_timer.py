import time
import numpy as np


class LapTimer:
    def __init__(self, max_laps: int = 100):
        self.max_laps = max_laps
        self.reset()

    def reset(self):
        self.lap_start_time = time.time()
        self.lap_times = []
        self.lap_count = 0
        self.segment_times = []

    def record_lap(self) -> float:
        elapsed = time.time() - self.lap_start_time
        self.lap_times.append(elapsed)
        self.lap_count += 1
        self.lap_start_time = time.time()
        return elapsed

    def record_segment(self, segment_name: str = ""):
        now = time.time()
        if self.segment_times:
            delta = now - self.segment_times[-1]["time"]
        else:
            delta = now - self.lap_start_time
        self.segment_times.append({"name": segment_name, "time": now, "delta": delta})

    @property
    def average_lap_time(self) -> float:
        return float(np.mean(self.lap_times)) if self.lap_times else 0.0

    @property
    def best_lap_time(self) -> float:
        return float(np.min(self.lap_times)) if self.lap_times else 0.0

    @property
    def worst_lap_time(self) -> float:
        return float(np.max(self.lap_times)) if self.lap_times else 0.0

    @property
    def lap_time_std(self) -> float:
        return float(np.std(self.lap_times)) if len(self.lap_times) > 1 else 0.0

    def summary(self) -> str:
        lines = [f"{'='*50}", f"{'Lap Time Summary':^50}", f"{'='*50}"]
        lines.append(f"{'Total Laps':25s} {self.lap_count}")
        if self.lap_times:
            lines.append(f"{'Average Lap':25s} {self.average_lap_time:.3f}s")
            lines.append(f"{'Best Lap':25s} {self.best_lap_time:.3f}s")
            lines.append(f"{'Worst Lap':25s} {self.worst_lap_time:.3f}s")
            lines.append(f"{'Std Dev':25s} {self.lap_time_std:.4f}s")
            lines.append(f"{'Total Time':25s} {sum(self.lap_times):.3f}s")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "lap_count": self.lap_count,
            "average_lap": round(self.average_lap_time, 3),
            "best_lap": round(self.best_lap_time, 3),
            "worst_lap": round(self.worst_lap_time, 3),
            "std": round(self.lap_time_std, 4),
            "all_laps": [round(t, 3) for t in self.lap_times],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    args = parser.parse_args()

    import os, yaml, torch
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

    timer = LapTimer()
    print(f"Timing {args.episodes} episodes...")
    for ep in range(args.episodes):
        state = env.reset()
        timer.reset()
        done = False
        while not done:
            action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
            state, _, done, info = env.step(action)
            if info.get("lap_count", 0) > timer.lap_count:
                timer.record_lap()
        print(f"  Ep {ep + 1}: {timer.lap_count} laps, best {timer.best_lap_time:.2f}s")
    print(timer.summary())


if __name__ == "__main__":
    main()
