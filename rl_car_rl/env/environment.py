import math
import numpy as np
from env.car import Car
from env.track import Track
from env.sensors import SensorSystem
from env.physics import PhysicsEngine

class CarEnv:
    def __init__(self):
        self.car = Car()
        self.track = Track()
        self.physics = PhysicsEngine()
        self.sensors = SensorSystem(num_sensors=5, max_distance=150.0)
        self.max_steps = 1000
        self.current_step = 0
        self.dt = 0.1
        self.previous_progress = 0.0
        self.previous_steering = 0.0
        self.track_params = {}
        
    def set_difficulty(self, params):
        self.track_params = params
        
    def reset(self):
        self.track.generate(**self.track_params)
        start = self.track.start_pose
        self.car.reset(start["x"], start["y"], start["heading"])
        self.current_step = 0
        self.previous_progress = self.track.get_distance_along_track(self.car.x, self.car.y)
        self.previous_steering = 0.0
        return self._get_observation()
        
    def _get_observation(self):
        # Phase 3 - Observation System
        # Combine sensor readings into state vector
        sensor_readings = self.sensors.get_readings(self.car, self.track)
        
        car_state = self.car.get_state()
        
        # Normalize inputs for neural network use
        normalized_velocity = car_state["velocity"] / self.physics.max_speed
        
        # heading angle is -pi to pi, so dividing by pi normalizes to [-1, 1]
        normalized_heading = car_state["heading"] / math.pi
        
        # angular velocity normalization heuristic
        normalized_angular_vel = car_state["angular_velocity"] / 2.0
        
        # Distance from center estimation
        # We can approximate track center distance if we treat coordinates 400,300 as center
        dx = car_state["x"] - 400
        dy = car_state["y"] - 300
        dist_from_center = math.hypot(dx, dy)
        max_possible_dist = 400.0 # From center to edges approx 400
        normalized_center_dist = min(dist_from_center / max_possible_dist, 1.0)
        
        # Size: 5 + 4 = 9
        # Example: [sensor1 .. sensor5, velocity, heading_angle, angular_velocity, distance_from_center]
        obs = sensor_readings + [
            normalized_velocity,
            normalized_heading,
            normalized_angular_vel,
            normalized_center_dist
        ]
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        action: [steering, throttle]
        """
        steering = float(action[0])
        throttle = float(action[1])
        
        # Phase 2 Physics update
        self.physics.update(self.car, steering, throttle, self.dt)
        self.current_step += 1
        
        obs = self._get_observation()
        
        # We'll use a placeholder reward structure to be updated in Phase 5
        # The prompt says I must implement reward calculation in Phase 5... but the Environment step
        # needs to return reward. I will import the reward function from training.
        from training.reward_function import compute_reward
        
        # Calculate progress
        current_progress = self.track.get_distance_along_track(self.car.x, self.car.y)
        progress_diff = current_progress - self.previous_progress
        if progress_diff < -0.5:
            progress_diff += 1.0
        self.previous_progress = current_progress
        
        # Collision checking
        corners = self.car.get_corners()
        crashed = self.track.check_collision(corners)
        
        done = crashed or self.current_step >= self.max_steps
        
        # Compute center distance for reward
        center_distance = self.track.get_center_distance(self.car.x, self.car.y)
        
        # Compute steering change for smoothness penalty
        steering_change = abs(float(steering) - self.previous_steering)
        self.previous_steering = float(steering)
        
        reward = compute_reward(
            self.car.velocity, progress_diff, crashed, done,
            center_distance=center_distance,
            steering_change=steering_change,
        )
        
        info = {
            "x": self.car.x,
            "y": self.car.y,
            "velocity": self.car.velocity,
            "crashed": crashed,
            "center_distance": center_distance,
        }
        
        return obs, reward, done, info
        
    def render(self):
        # Will be implemented in Phase 7 properly. But since tasks.md says we must have a render hook:
        pass
