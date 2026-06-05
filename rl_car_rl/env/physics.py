import math
from env.car import Car


class PhysicsEngine:
    def __init__(self):
        self.max_speed = 50.0
        self.max_steering_angle = math.pi / 4  # 45 degrees
        self.acceleration_factor = 20.0
        self.drag_coefficient = 0.005
        self.tire_grip = 1.0
        self.traction_circle_radius = 1.0

    def update(
        self,
        car: Car,
        steering_input: float,
        throttle_input: float,
        dt: float = 0.1,
    ) -> None:
        """Enhanced physics with slip angle, traction circle coupling, and drag."""
        steering = max(min(steering_input, 1.0), -1.0)
        throttle = max(min(throttle_input, 1.0), 0.0)

        car.steering_angle = steering * self.max_steering_angle
        car.throttle = throttle

        # Lateral force (steering demand)
        lateral_force = abs(steering) * self.tire_grip

        # Longitudinal force (throttle demand)
        longitudinal_force = throttle

        # Traction circle: combined force is limited
        # High steering reduces available acceleration, and vice versa
        if lateral_force > 0.01 or longitudinal_force > 0.01:
            combined = math.hypot(lateral_force, longitudinal_force)
            if combined > self.traction_circle_radius:
                scale = self.traction_circle_radius / combined
                lateral_force *= scale
                longitudinal_force *= scale

        # Drag force (quadratic with velocity)
        drag = self.drag_coefficient * car.velocity * abs(car.velocity)

        # Net acceleration
        net_accel = longitudinal_force * self.acceleration_factor - drag
        car.velocity += net_accel * dt

        # Clamp velocity
        car.velocity = max(min(car.velocity, self.max_speed), -self.max_speed * 0.5)

        # Steering / turning
        if abs(car.velocity) < 0.1:
            car.angular_velocity = 0.0
        else:
            effective_steering = car.steering_angle * lateral_force
            car.angular_velocity = (
                car.velocity / car.length
            ) * math.tan(effective_steering)

            # Clamp angular velocity
            max_ang_vel = abs(car.velocity) / (car.length * 2)
            car.angular_velocity = max(
                min(car.angular_velocity, max_ang_vel), -max_ang_vel
            )

        # Update heading
        car.heading += car.angular_velocity * dt
        car.heading = (car.heading + math.pi) % (2 * math.pi) - math.pi

        # Update position
        car.x += car.velocity * math.cos(car.heading) * dt
        car.y += car.velocity * math.sin(car.heading) * dt

