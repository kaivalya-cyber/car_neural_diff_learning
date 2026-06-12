import numpy as np


class AdaptiveNoiseScheduler:
    def __init__(self, initial_sigma: float = 0.2, min_sigma: float = 0.01,
                 decay_rate: float = 0.995, plateau_window: int = 20,
                 plateau_threshold: float = 0.05, noise_increase_factor: float = 1.2):
        self.sigma = initial_sigma
        self.min_sigma = min_sigma
        self.decay_rate = decay_rate
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.noise_increase_factor = noise_increase_factor
        self.recent_rewards = []
        self.last_plateau_reset = 0

    def step(self, reward: float):
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.plateau_window:
            self.recent_rewards.pop(0)

        if len(self.recent_rewards) >= self.plateau_window:
            first_half = np.mean(self.recent_rewards[:self.plateau_window // 2])
            second_half = np.mean(self.recent_rewards[self.plateau_window // 2:])
            improvement = abs(second_half - first_half) / (abs(first_half) + 1e-8)

            if improvement < self.plateau_threshold:
                self.sigma = min(self.sigma * self.noise_increase_factor, 0.5)
                self.last_plateau_reset = len(self.recent_rewards)
            else:
                self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)
        else:
            self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)

    def get_sigma(self) -> float:
        return self.sigma

    def get_noise_params(self) -> dict:
        return {"sigma": self.sigma, "theta": 0.15, "sigma_decay": 1.0}

    def reset(self, sigma: float | None = None):
        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = 0.2
        self.recent_rewards = []
        self.last_plateau_reset = 0


class LinearNoiseSchedule:
    def __init__(self, start_sigma: float = 0.3, end_sigma: float = 0.01, total_steps: int = 50000):
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.total_steps = total_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        frac = min(self.step_count / self.total_steps, 1.0)
        self.sigma = self.start_sigma + (self.end_sigma - self.start_sigma) * frac

    def get_sigma(self) -> float:
        frac = min(self.step_count / self.total_steps, 1.0)
        return self.start_sigma + (self.end_sigma - self.start_sigma) * frac


def main():
    import time
    scheduler = AdaptiveNoiseScheduler(initial_sigma=0.2)
    print("Adaptive Noise Scheduler Simulation")
    print("=" * 50)
    for i in range(200):
        mock_reward = 30 + 10 * np.sin(i / 20) + np.random.randn() * 5
        scheduler.step(mock_reward)
        if i % 20 == 0:
            print(f"  Step {i:4d} | Reward: {mock_reward:6.1f} | Sigma: {scheduler.get_sigma():.4f}")
    print(f"\nFinal sigma: {scheduler.get_sigma():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
