import numpy as np


class RewardBreakdown:
    def __init__(self):
        self.reset()

    def reset(self):
        self.components = {
            "progress": [],
            "speed_zone": [],
            "centerline": [],
            "smoothness": [],
            "time_penalty": [],
            "crash": [],
        }
        self.step_count = 0

    def log_step(self, progress=0.0, speed=0.0, center_dist=0.0, steering=0.0, crashed=False):
        self.step_count += 1
        self.components["progress"].append(50.0 * progress)
        optimal = 20.0 < speed < 40.0
        self.components["speed_zone"].append(10.0 if optimal else 0.0)
        self.components["centerline"].append(max(0, 10.0 - center_dist * 2))
        self.components["smoothness"].append(-abs(steering) * 0.5)
        self.components["time_penalty"].append(-0.1)
        self.components["crash"].append(-10.0 if crashed else 0.0)

    def totals(self) -> dict:
        return {k: sum(v) for k, v in self.components.items()}

    def report(self) -> str:
        totals = self.totals()
        total = sum(totals.values())
        lines = [f"{'Reward Component':25s} {'Total':>10s} {'%':>8s}"]
        lines.append("-" * 45)
        for key in self.components:
            val = totals[key]
            pct = (val / total * 100) if total != 0 else 0
            lines.append(f"{key:25s} {val:>10.2f} {pct:>7.1f}%")
        lines.append("-" * 45)
        lines.append(f"{'TOTAL':25s} {total:>10.2f} {'100.0%':>8s}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"components": self.totals(), "total": sum(self.totals().values()), "steps": self.step_count}


def analyze_logged_rewards(csv_path: str = "logs/metrics.csv") -> dict:
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        return {
            "episodes": len(df),
            "mean_reward": float(df["reward"].mean()),
            "std_reward": float(df["reward"].std()),
            "max_reward": float(df["reward"].max()),
            "min_reward": float(df["reward"].min()),
            "mean_length": float(df["length"].mean()),
            "crash_rate": float(df["crash_rate"].mean()),
            "mean_laps": float(df["laps"].mean()),
            "best_episode": int(df["reward"].idxmax()) + 1 if not df.empty else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/metrics.csv"
    stats = analyze_logged_rewards(path)
    print("=" * 50)
    print("Training Reward Analysis")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
