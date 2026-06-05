import math
import numpy as np
from env.car import Car
from env.track import Track
from env.sensors import SensorSystem
from env.physics import PhysicsEngine


class CarEnv:
    def __init__(self, sensor_count: int = 16, sensor_max_distance: float = 150.0, obstacle_count: int = 0, track_type: str = "procedural", action_repeat: int = 1):
        self.car = Car()
        self.track = Track()
        self.physics = PhysicsEngine()
        self.sensors = SensorSystem(
            sensor_count=sensor_count,
            max_distance=sensor_max_distance,
            spread_degrees=360.0,
        )
        self.sensor_count = sensor_count
        self.obstacle_count = obstacle_count
        self.track_type = track_type
        self.action_repeat = action_repeat
        self.state_dim = sensor_count + 4
        self.max_steps = 1000
        self.current_step = 0
        self.dt = 0.1
        self.previous_progress = 0.0
        self.previous_steering = 0.0
        self.track_params = {}
        self.lap_count = 0
        self._crossed_start = False

    def set_difficulty(self, params: dict) -> None:
        self.track_params = params

    def reset(self) -> np.ndarray:
        self.track.generate(obstacle_count=self.obstacle_count, track_type=self.track_type, **self.track_params)
        start = self.track.start_pose
        self.car.reset(start["x"], start["y"], start["heading"])
        self.current_step = 0
        self.previous_progress = self.track.get_distance_along_track(
            self.car.x, self.car.y
        )
        self.previous_steering = 0.0
        self.lap_count = 0
        self._crossed_start = False
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        sensor_readings = self.sensors.get_readings(self.car, self.track)
        car_state = self.car.get_state()

        # Normalize inputs for neural network use
        normalized_velocity = car_state["velocity"] / self.physics.max_speed

        # heading angle is -pi to pi, so dividing by pi normalizes to [-1, 1]
        normalized_heading = car_state["heading"] / math.pi

        # angular velocity normalization heuristic
        normalized_angular_vel = car_state["angular_velocity"] / 2.0

        # Distance from centerline
        center_distance = self.track.get_center_distance(self.car.x, self.car.y)
        # Already normalized [0, 1] from get_center_distance

        # Observation: [sensor0..sensorN, velocity, heading, angular_vel, center_dist]
        obs = sensor_readings + [
            normalized_velocity,
            normalized_heading,
            normalized_angular_vel,
            center_distance,
        ]

        return np.array(obs, dtype=np.float32)

    def step(
        self, action: list[float] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step (with optional action repeat).

        Args:
            action: [steering, throttle]

        Returns:
            obs, reward, done, info
        """
        steering = float(action[0])
        throttle = float(action[1])

        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.action_repeat):
            if done:
                break

            # Physics update
            self.physics.update(self.car, steering, throttle, self.dt)
            self.current_step += 1

            # Calculate progress
            current_progress = self.track.get_distance_along_track(
                self.car.x, self.car.y
            )
            progress_diff = current_progress - self.previous_progress

            # Detect lap completion
            if progress_diff < -0.5:
                self.lap_count += 1
                self._crossed_start = True
                progress_diff += 1.0
            elif progress_diff > 0.5:
                self.lap_count = max(0, self.lap_count - 1)
                self._crossed_start = False
                progress_diff -= 1.0

            self.previous_progress = current_progress

            corners = self.car.get_corners()
            boundary_crash = self.track.check_collision(corners)
            obstacle_crash = self.track.check_obstacle_collision(corners)
            crashed = boundary_crash or obstacle_crash

            done = crashed or self.current_step >= self.max_steps

            center_distance = self.track.get_center_distance(self.car.x, self.car.y)
            steering_change = abs(float(steering) - self.previous_steering)
            self.previous_steering = float(steering)

            from training.reward_function import compute_reward
            step_reward = compute_reward(
                self.car.velocity,
                progress_diff,
                crashed,
                done,
                center_distance=center_distance,
                steering_change=steering_change,
            )

            if self._crossed_start and not done:
                step_reward += 50.0
                self._crossed_start = False

            total_reward += step_reward

        obs = self._get_observation()

        info = {
            "x": self.car.x,
            "y": self.car.y,
            "velocity": self.car.velocity,
            "crashed": crashed,
            "obstacle_crash": obstacle_crash,
            "center_distance": center_distance,
            "lap_count": self.lap_count,
            "progress": self.previous_progress,
            "obstacles": len(self.track.obstacles),
        }

        return obs, total_reward, done, info

    def render(self) -> None:
        pass
