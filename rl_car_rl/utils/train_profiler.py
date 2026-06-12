import time
import numpy as np


class TrainProfiler:
    def __init__(self):
        self.timings = {
            "env_step": [],
            "policy_inference": [],
            "memory_append": [],
            "ppo_update": [],
            "curriculum_update": [],
            "validation": [],
            "logging": [],
            "checkpoint_save": [],
        }
        self._current_section = None
        self._start_time = None

    def start(self, section: str):
        self._current_section = section
        self._start_time = time.perf_counter()

    def stop(self):
        if self._current_section and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            if self._current_section in self.timings:
                self.timings[self._current_section].append(elapsed)
        self._current_section = None
        self._start_time = None

    def summary(self) -> str:
        lines = [f"{'='*60}", f"{'Training Profiler Summary':^60}", f"{'='*60}"]
        lines.append(f"{'Section':25s} {'Count':8s} {'Total (s)':12s} {'Mean (ms)':12s}")
        lines.append(f"{'-'*60}")

        total_time = 0
        for section, times in self.timings.items():
            if times:
                count = len(times)
                total = sum(times)
                mean_ms = (total / count) * 1000
                total_time += total
                lines.append(f"{section:25s} {count:<8d} {total:<12.4f} {mean_ms:<12.2f}")

        lines.append(f"{'-'*60}")
        lines.append(f"{'TOTAL':25s} {'':8s} {total_time:<12.4f}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def profile_epoch(self, env_step_fn, policy_fn, update_fn=None, num_steps: int = 100) -> dict:
        for _ in range(num_steps):
            self.start("env_step")
            state, reward, done, info = env_step_fn()
            self.stop()

            self.start("policy_inference")
            action = policy_fn(state)
            self.stop()

        if update_fn:
            self.start("ppo_update")
            update_fn()
            self.stop()

        return {k: {"count": len(v), "total": sum(v), "mean_ms": np.mean(v) * 1000 if v else 0}
                for k, v in self.timings.items() if v}

    def reset(self):
        for k in self.timings:
            self.timings[k].clear()


def main():
    profiler = TrainProfiler()

    def mock_env_step():
        time.sleep(0.005)
        return np.zeros(20), 1.0, False, {}

    def mock_policy(s):
        time.sleep(0.003)
        return np.array([0.0, 1.0])

    def mock_update():
        time.sleep(0.05)

    import numpy as np
    results = profiler.profile_epoch(mock_env_step, mock_policy, mock_update, num_steps=50)
    print(profiler.summary())


if __name__ == "__main__":
    main()
