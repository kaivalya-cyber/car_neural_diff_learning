import time
import shutil
from collections import deque


class ConsoleDashboard:
    def __init__(self, window_size: int = 100, update_interval: float = 1.0):
        self.window_size = window_size
        self.update_interval = update_interval
        self.metrics = {
            "episode": 0,
            "reward": deque(maxlen=window_size),
            "length": deque(maxlen=window_size),
            "crash_rate": deque(maxlen=window_size),
            "difficulty": deque(maxlen=window_size),
            "fps": deque(maxlen=100),
            "lr": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }
        self._last_update = time.time()
        self._step_counter = 0
        self._last_step_counter = 0
        self._last_fps_time = time.time()

    def update(self, **kwargs):
        now = time.time()
        self._step_counter += 1

        for k, v in kwargs.items():
            if k in self.metrics and isinstance(self.metrics[k], deque):
                self.metrics[k].append(v)
            elif k in self.metrics:
                self.metrics[k] = v

        if now - self._last_fps_time >= 1.0:
            steps = self._step_counter - self._last_step_counter
            self.metrics["fps"].append(steps / max(now - self._last_fps_time, 0.001))
            self._last_step_counter = self._step_counter
            self._last_fps_time = now

        if now - self._last_update >= self.update_interval:
            self._render()
            self._last_update = now

    def _render(self):
        cols = shutil.get_terminal_size((80, 24)).columns
        width = min(cols, 100)

        print()
        print("=" * width)
        ep = self.metrics["episode"]
        print(f"  Training Progress  |  Episode: {ep}")
        print("-" * width)

        if self.metrics["reward"]:
            r_arr = list(self.metrics["reward"])
            print(f"  Reward:     {r_arr[-1]:>8.2f}  |  Avg: {sum(r_arr[-50:])/len(r_arr[-50:]):>8.2f}  |  Best: {max(r_arr):>8.2f}")
        if self.metrics["length"]:
            l_arr = list(self.metrics["length"])
            print(f"  Length:     {l_arr[-1]:>8d}  |  Avg: {sum(l_arr[-50:])/len(l_arr[-50:]):>7.1f}")
        if self.metrics["crash_rate"]:
            cr = list(self.metrics["crash_rate"])
            print(f"  Crash Rate: {cr[-1]:>8.2%}  |  Avg: {sum(cr[-50:])/len(cr[-50:]):>8.2%}")
        if self.metrics["difficulty"]:
            d_arr = list(self.metrics["difficulty"])
            print(f"  Difficulty: {d_arr[-1]:>8.3f}")
        if self.metrics["fps"]:
            fps_arr = list(self.metrics["fps"])
            print(f"  FPS:        {fps_arr[-1]:>8.1f}")
        print(f"  LR:         {self.metrics['lr']:>8.2e}  |  Entropy: {self.metrics['entropy']:.4f}")
        print(f"  Policy Loss:{self.metrics['policy_loss']:>8.4f}  |  Value Loss: {self.metrics['value_loss']:>7.4f}")
        print("=" * width)

    def progress_bar(self, current: int, total: int, bar_width: int = 40) -> str:
        frac = current / max(total, 1)
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)
        return f"|{bar}| {current}/{total} ({frac:.1%})"


def main():
    import time
    import random

    dash = ConsoleDashboard(update_interval=0.5)
    print("Console Dashboard Demo (10 seconds)")
    print("Press Ctrl+C to stop")

    try:
        for ep in range(50):
            reward = random.uniform(10, 80)
            length = random.randint(100, 1000)
            crashed = random.random() < 0.3
            dash.update(
                episode=ep,
                reward=reward,
                length=length,
                crash_rate=float(crashed),
                difficulty=random.uniform(0, 1),
                lr=3e-4 * (0.99 ** ep),
                entropy=0.01 * (0.995 ** ep),
                policy_loss=random.uniform(0.001, 0.1),
                value_loss=random.uniform(0.01, 0.5),
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
