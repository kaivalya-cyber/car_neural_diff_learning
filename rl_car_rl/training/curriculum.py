import numpy as np
from collections import deque


class CurriculumManager:
    def __init__(
        self,
        start_level: float = 0.0,
        min_level: float = 0.0,
        max_level: float = 1.0,
        increase_threshold: float = 40.0,
        decrease_threshold: float = 10.0,
        increase_rate: float = 0.05,
        decrease_rate: float = 0.03,
        threshold_growth: float = 1.01,
        window_size: int = 10,
        min_samples: int = 5,
    ):
        """
        Manages the difficulty level of track generation with bidirectional
        adjustment and hysteresis to prevent oscillation.

        Args:
            start_level: Initial difficulty level [0, 1].
            min_level: Floor for difficulty (prevents going below this).
            max_level: Ceiling for difficulty.
            increase_threshold: Mean reward required to increase difficulty.
            decrease_threshold: Mean reward below which difficulty decreases.
            increase_rate: How much to increase level per successful evaluation.
            decrease_rate: How much to decrease level per poor evaluation.
            threshold_growth: Factor by which the increase_threshold grows after
                each successful increase (>1 makes advancement harder).
            window_size: Number of recent rewards to average for decisions.
            min_samples: Minimum samples needed before making curriculum decisions.
        """
        self.level = start_level
        self.min_level = min_level
        self.max_level = max_level
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.threshold_growth = threshold_growth
        self.window_size = window_size
        self.min_samples = min_samples
        self.recent_rewards = deque(maxlen=window_size)
        self.total_updates = 0
        self.last_direction = 0  # -1=decreased, 0=none, 1=increased (for hysteresis)

        # Difficulty bounds for generation params
        # (width, min_radius, max_radius, num_control_points)
        self.easy_params = (120, 300, 350, 8)
        self.hard_params = (60, 100, 450, 16)

    def add_reward(self, reward: float) -> None:
        """Add a reward sample to the rolling window."""
        self.recent_rewards.append(reward)

    def update(self, mean_reward: float | None = None) -> float:
        """
        Updates the curriculum level based on recent performance.
        If mean_reward is provided, it's added to the window first.

        Returns the new curriculum level.
        """
        if mean_reward is not None:
            self.add_reward(mean_reward)

        self.total_updates += 1

        if len(self.recent_rewards) < self.min_samples:
            return self.level

        rolling_mean = np.mean(self.recent_rewards)
        old_level = self.level

        # Bidirectional adjustment with hysteresis
        if rolling_mean >= self.increase_threshold:
        # Good performance: increase difficulty
            self.level = min(self.max_level, self.level + self.increase_rate)
            # Grow threshold to make further advancement harder
            self.increase_threshold *= self.threshold_growth
            self.last_direction = 1

        elif rolling_mean <= self.decrease_threshold:
            # Poor performance: decrease difficulty (but not below min_level)
            if self.level > self.min_level:
                # Only decrease if we didn't just increase (hysteresis)
                if self.last_direction != 1:
                    self.level = max(self.min_level, self.level - self.decrease_rate)
                    self.last_direction = -1
                else:
                    # Just increased — give it more time, reset direction
                    self.last_direction = 0

        else:
            # Moderate performance: hold steady
            self.last_direction = 0

        return self.level

    def get_generation_params(self) -> dict:
        """
        Returns interpolated track generation parameters for the current level.
        Uses smoothstep interpolation for more natural progression.
        """
        # Smoothstep for more natural transition
        t = self.level  # [0, 1]
        # Smoothstep: 3t^2 - 2t^3
        t_smooth = 3.0 * t * t - 2.0 * t * t * t

        ew, eminr, emaxr, encp = self.easy_params
        hw, hminr, hmaxr, hncp = self.hard_params

        w = ew + (hw - ew) * t_smooth
        minr = eminr + (hminr - eminr) * t_smooth
        maxr = emaxr + (hmaxr - emaxr) * t_smooth
        ncp = int(encp + (hncp - encp) * t_smooth)

        return {
            "track_width": w,
            "min_radius": minr,
            "max_radius": maxr,
            "num_control_points": ncp,
        }

    def get_stats(self) -> dict:
        """Return curriculum statistics for logging."""
        return {
            "level": self.level,
            "increase_threshold": self.increase_threshold,
            "decrease_threshold": self.decrease_threshold,
            "rolling_mean": (
                float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
            ),
            "window_size": len(self.recent_rewards),
            "total_updates": self.total_updates,
        }
