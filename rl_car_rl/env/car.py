import math

class Car:
    def __init__(self, init_x: float = 0.0, init_y: float = 0.0, init_heading: float = 0.0):
        self.x = init_x
        self.y = init_y
        self.heading = init_heading
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0
        
        # Dimensions
        self.length = 40.0
        self.width = 20.0

    def reset(self, x: float, y: float, heading: float):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
            "steering": self.steering_angle,
            "throttle": self.throttle
        }

    def get_corners(self):
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        
        half_l = self.length / 2
        half_w = self.width / 2

        corners = [
            (half_l, half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
            (half_l, -half_w)
        ]

        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_h - cy * sin_h + self.x
            ry = cx * sin_h + cy * cos_h + self.y
            rotated_corners.append((rx, ry))
            
        return rotated_corners
