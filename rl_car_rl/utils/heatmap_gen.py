import numpy as np
import os


class TrackHeatmapper:
    def __init__(self, grid_size: tuple[int, int] = (100, 100)):
        self.grid_size = grid_size
        self.occupancy = np.zeros(grid_size, dtype=np.float32)
        self.crash_map = np.zeros(grid_size, dtype=np.float32)
        self.speed_map = np.zeros(grid_size, dtype=np.float32)
        self.total_visits = np.zeros(grid_size, dtype=np.int32)

    def _to_grid(self, x: float, y: float, bounds: tuple) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = bounds
        gx = int((x - x_min) / (x_max - x_min + 1e-8) * (self.grid_size[0] - 1))
        gy = int((y - y_min) / (y_max - y_min + 1e-8) * (self.grid_size[1] - 1))
        return max(0, min(gx, self.grid_size[0] - 1)), max(0, min(gy, self.grid_size[1] - 1))

    def record_position(self, x: float, y: float, bounds: tuple, speed: float = 0.0, crashed: bool = False):
        gx, gy = self._to_grid(x, y, bounds)
        self.occupancy[gy, gx] += 1.0
        self.speed_map[gy, gx] += speed
        self.total_visits[gy, gx] += 1
        if crashed:
            self.crash_map[gy, gx] += 1.0

    def normalize(self):
        mask = self.total_visits > 0
        self.occupancy = self.occupancy / (self.occupancy.max() + 1e-8)
        self.speed_map[mask] = self.speed_map[mask] / self.total_visits[mask]
        self.crash_map[mask] = self.crash_map[mask] / self.total_visits[mask]

    def get_heatmap_data(self) -> dict:
        self.normalize()
        return {
            "occupancy": self.occupancy.tolist(),
            "crash_rate": self.crash_map.tolist(),
            "avg_speed": self.speed_map.tolist(),
            "visits": self.total_visits.tolist(),
        }

    def save_npz(self, path: str = "heatmap_data.npz"):
        self.normalize()
        np.savez(path, occupancy=self.occupancy, crash_rate=self.crash_map,
                 avg_speed=self.speed_map, visits=self.total_visits)
        print(f"Saved heatmap data to {path}")


def collect_and_save(env, trainer, num_episodes: int = 50, output: str = "heatmap_data.npz"):
    bounds = (0, 0, env.track.track_width, env.track.track_height)
    heatmapper = TrackHeatmapper()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            heatmapper.record_position(
                env.car.x, env.car.y, bounds,
                speed=env.car.velocity, crashed=info.get("crashed", False)
            )
        if (ep + 1) % 10 == 0:
            print(f"  Collected {ep + 1}/{num_episodes} episodes")

    heatmapper.save_npz(output)
    return heatmapper


def main():
    import argparse, sys, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output", default="heatmap_data.npz")
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

    collect_and_save(env, trainer, args.episodes, args.output)


if __name__ == "__main__":
    main()
