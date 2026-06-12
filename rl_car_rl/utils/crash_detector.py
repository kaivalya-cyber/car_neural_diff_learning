import numpy as np
from collections import deque


class EarlyCrashDetector:
    def __init__(self, window_size: int = 5, proximity_threshold: float = 0.15,
                 speed_drop_threshold: float = 0.3, angular_spike_threshold: float = 0.8):
        self.window_size = window_size
        self.proximity_threshold = proximity_threshold
        self.speed_drop_threshold = speed_drop_threshold
        self.angular_spike_threshold = angular_spike_threshold
        self.recent_proximities = deque(maxlen=window_size)
        self.recent_speeds = deque(maxlen=window_size)
        self.recent_steering = deque(maxlen=window_size)
        self.crash_warnings = 0

    def update(self, sensor_readings: np.ndarray, velocity: float, steering: float) -> dict:
        min_dist = float(sensor_readings.min()) if len(sensor_readings) > 0 else 1.0
        self.recent_proximities.append(min_dist)
        self.recent_speeds.append(velocity)
        self.recent_steering.append(abs(steering))

        signals = {
            "proximity_warning": False,
            "speed_drop_warning": False,
            "angular_spike_warning": False,
            "crash_risk": 0.0,
        }

        if len(self.recent_proximities) >= self.window_size:
            avg_prox = np.mean(self.recent_proximities)
            if avg_prox < self.proximity_threshold:
                signals["proximity_warning"] = True

        if len(self.recent_speeds) >= 3:
            speed_drop = (self.recent_speeds[-3] - self.recent_speeds[-1]) / (self.recent_speeds[-3] + 1e-8)
            if speed_drop > self.speed_drop_threshold:
                signals["speed_drop_warning"] = True

        if len(self.recent_steering) >= 3:
            if np.mean(self.recent_steering) > self.angular_spike_threshold:
                signals["angular_spike_warning"] = True

        risk = sum([
            signals["proximity_warning"],
            signals["speed_drop_warning"],
            signals["angular_spike_warning"],
        ]) / 3.0
        signals["crash_risk"] = risk

        if risk > 0.5:
            self.crash_warnings += 1

        return signals

    def predict_crash(self, threshold: float = 0.6) -> bool:
        return self.crash_warnings > 0

    def reset(self):
        self.recent_proximities.clear()
        self.recent_speeds.clear()
        self.recent_steering.clear()
        self.crash_warnings = 0


def main():
    detector = EarlyCrashDetector()
    print("Crash Detector Simulation")
    print("=" * 50)

    for step in range(50):
        readings = np.random.rand(16)
        if step > 30:
            readings[:5] = np.random.rand(5) * 0.1
        vel = max(0, 30 - max(0, step - 25) * 2)
        steer = 0.3 + max(0, step - 28) * 0.05

        signals = detector.update(readings, vel, steer)
        if signals["crash_risk"] > 0.3:
            print(f"  Step {step:3d} | Risk: {signals['crash_risk']:.2f} | "
                  f"P:{signals['proximity_warning']} S:{signals['speed_drop_warning']} A:{signals['angular_spike_warning']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
