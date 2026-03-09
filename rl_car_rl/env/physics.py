import math
from env.car import Car

class PhysicsEngine:
    def __init__(self):
        self.max_speed = 50.0
        self.max_steering_angle = math.pi / 4  # 45 degrees
        self.acceleration_factor = 20.0
        self.friction = 0.95

    def update(self, car: Car, steering_input: float, throttle_input: float, dt: float = 0.1):
        """
        Applies bicycle model steering and velocity updates to the Car instance.
        steering_input: [-1, 1]
        throttle_input: [0, 1]
        """
        # Constrain inputs and store them
        steering = max(min(steering_input, 1.0), -1.0)
        throttle = max(min(throttle_input, 1.0), 0.0)
        
        car.steering_angle = steering * self.max_steering_angle
        car.throttle = throttle

        # Velocity update
        acceleration = car.throttle * self.acceleration_factor
        car.velocity += acceleration * dt
        car.velocity *= self.friction
        car.velocity = min(car.velocity, self.max_speed)

        # Bicycle model steering
        if abs(car.velocity) < 0.1:
            car.angular_velocity = 0.0
        else:
            car.angular_velocity = (car.velocity / car.length) * math.tan(car.steering_angle)
            car.heading += car.angular_velocity * dt

        # Normalize heading
        car.heading = (car.heading + math.pi) % (2 * math.pi) - math.pi

        # Timestep update for position
        car.x += car.velocity * math.cos(car.heading) * dt
        car.y += car.velocity * math.sin(car.heading) * dt
