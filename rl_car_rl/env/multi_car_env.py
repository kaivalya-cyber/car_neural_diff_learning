import math
import numpy as np
from env.car import Car
from env.track import Track
from env.sensors import SensorSystem
from env.physics import PhysicsEngine
from env.environment import CarEnv


class MultiCarEnv:
    """
    Multi-agent racing environment with multiple cars on the same track.
    Supports competitive racing with position-based rewards and car-to-car collision.
    """

    def __init__(
        self,
        num_cars: int = 2,
        sensor_count: int = 16,
        sensor_max_distance: float = 150.0,
        obstacle_count: int = 0,
    ):
        self.num_cars = num_cars
        self.sensor_count = sensor_count
        self.state_dim = sensor_count + 4

        # Shared track
        self.track = Track()
        self.physics = PhysicsEngine()

        # Create cars with different colors
        car_colors = [
            (0, 180, 0),    # green (car 0)
            (0, 100, 255),  # blue (car 1)
            (255, 180, 0),  # orange (car 2)
            (255, 60, 60),  # red (car 3)
        ]
        self.cars = [
            Car(color=car_colors[i % len(car_colors)], name=f"car_{i}")
            for i in range(num_cars)
        ]

        # Each car has its own sensor system
        self.sensors_list = [
            SensorSystem(
                sensor_count=sensor_count,
                max_distance=sensor_max_distance,
                spread_degrees=360.0,
            )
            for _ in range(num_cars)
        ]

        self.obstacle_count = obstacle_count
        self.max_steps = 1000
        self.current_step = 0
        self.dt = 0.1
        self.track_params = {}
        self.lap_counts = [0] * num_cars
        self._crossed_start = [False] * num_cars
        self.previous_progress = [0.0] * num_cars
        self.previous_steering = [0.0] * num_cars

    def set_difficulty(self, params: dict) -> None:
        self.track_params = params

    def reset(self) -> np.ndarray:
        """Reset environment. Returns stacked observations for all cars."""
        self.track.generate(
            obstacle_count=self.obstacle_count, **self.track_params
        )
        start = self.track.start_pose

        # Place cars at different starting positions
        for i, car in enumerate(self.cars):
            # Offset each car by a few centerline points
            offset_idx = (i * 10) % len(self.track.center_points)
            x, y = self.track.center_points[offset_idx]
            # Compute heading from tangent
            next_idx = (offset_idx + 1) % len(self.track.center_points)
            nx, ny = self.track.center_points[next_idx]
            heading = math.atan2(ny - y, nx - x)
            car.reset(x, y, heading)
            self.previous_progress[i] = self.track.get_distance_along_track(x, y)
            self.previous_steering[i] = 0.0
            self.lap_counts[i] = 0
            self._crossed_start[i] = False

        self.current_step = 0
        return self._get_observations()

    def _get_observations(self) -> np.ndarray:
        """Return stacked observations: shape (num_cars, state_dim)."""
        obs_list = []
        for i, car in enumerate(self.cars):
            sensor_readings = self.sensors_list[i].get_readings(car, self.track)
            car_state = car.get_state()

            normalized_velocity = car_state["velocity"] / self.physics.max_speed
            normalized_heading = car_state["heading"] / math.pi
            normalized_angular_vel = car_state["angular_velocity"] / 2.0
            center_distance = self.track.get_center_distance(car.x, car.y)

            obs = sensor_readings + [
                normalized_velocity,
                normalized_heading,
                normalized_angular_vel,
                center_distance,
            ]
            obs_list.append(obs)

        return np.array(obs_list, dtype=np.float32)

    def _check_car_collisions(self) -> list[bool]:
        """Check all car-to-car collisions. Returns list of collision flags."""
        crashed = [False] * self.num_cars
        for i in range(self.num_cars):
            for j in range(i + 1, self.num_cars):
                if self.cars[i].collides_with(self.cars[j]):
                    crashed[i] = True
                    crashed[j] = True
        return crashed

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """
        Execute one step for all cars.

        Args:
            actions: shape (num_cars, 2) — [steering, throttle] per car.

        Returns:
            obs: (num_cars, state_dim), rewards: (num_cars,), dones: (num_cars,),
            infos: list of dict per car
        """
        self.current_step += 1

        # Update all cars
        for i, car in enumerate(self.cars):
            steering = float(actions[i][0])
            throttle = float(actions[i][1])
            self.physics.update(car, steering, throttle, self.dt)

        obs = self._get_observations()

        # Check collisions
        car_crashes = self._check_car_collisions()

        # Compute rewards and track progress
        rewards = np.zeros(self.num_cars, dtype=np.float32)
        dones = np.zeros(self.num_cars, dtype=bool)
        infos = []

        from training.reward_function import compute_reward

        # Track positions (by lap_count * 1000 + progress)
        positions = []
        for i, car in enumerate(self.cars):
            # Collision checking
            corners = car.get_corners()
            boundary_crash = self.track.check_collision(corners)
            obstacle_crash = self.track.check_obstacle_collision(corners)
            crashed = boundary_crash or obstacle_crash or car_crashes[i]

            # Progress tracking
            current_progress = self.track.get_distance_along_track(car.x, car.y)
            progress_diff = current_progress - self.previous_progress[i]

            if progress_diff < -0.5:
                self.lap_counts[i] += 1
                self._crossed_start[i] = True
                progress_diff += 1.0
            elif progress_diff > 0.5:
                self.lap_counts[i] = max(0, self.lap_counts[i] - 1)
                self._crossed_start[i] = False
                progress_diff -= 1.0

            self.previous_progress[i] = current_progress

            # Position metric for competitive reward
            position_score = self.lap_counts[i] * 1.0 + current_progress
            positions.append((position_score, i))

            # Steering smoothness
            steering_change = abs(float(actions[i][0]) - self.previous_steering[i])
            self.previous_steering[i] = float(actions[i][0])

            # Center distance
            center_distance = self.track.get_center_distance(car.x, car.y)

            done = crashed or self.current_step >= self.max_steps
            dones[i] = done

            reward = compute_reward(
                car.velocity,
                progress_diff,
                crashed,
                done,
                center_distance=center_distance,
                steering_change=steering_change,
            )

            # Lap completion bonus
            if self._crossed_start[i] and not done:
                reward += 50.0
                self._crossed_start[i] = False

            # Car-to-car collision penalty
            if car_crashes[i]:
                reward -= 5.0

            rewards[i] = reward

            infos.append(
                {
                    "x": car.x,
                    "y": car.y,
                    "velocity": car.velocity,
                    "crashed": crashed,
                    "car_crash": car_crashes[i],
                    "boundary_crash": boundary_crash,
                    "obstacle_crash": obstacle_crash,
                    "center_distance": center_distance,
                    "lap_count": self.lap_counts[i],
                    "progress": current_progress,
                    "obstacles": len(self.track.obstacles),
                }
            )

        # Competitive reward: position-based bonus
        positions.sort(key=lambda x: x[0], reverse=True)  # descending
        for rank, (_, car_idx) in enumerate(positions):
            if rank == 0:
                # Leader bonus
                rewards[car_idx] += 0.5
            elif rank == len(positions) - 1:
                # Last place small penalty
                rewards[car_idx] -= 0.2

        return obs, rewards, dones, infos

    def render(self) -> None:
        pass
