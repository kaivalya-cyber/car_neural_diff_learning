import math


class Car:
    def __init__(
        self,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_heading: float = 0.0,
        color: tuple[int, int, int] = (0, 180, 0),
        name: str = "",
    ):
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

        # Visual identity
        self.color = color
        self.name = name

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.steering_angle = 0.0
        self.throttle = 0.0

    def get_state(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
            "steering": self.steering_angle,
            "throttle": self.throttle,
        }

    def get_corners(self) -> list[tuple[float, float]]:
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        half_l = self.length / 2
        half_w = self.width / 2

        corners = [
            (half_l, half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
            (half_l, -half_w),
        ]

        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_h - cy * sin_h + self.x
            ry = cx * sin_h + cy * cos_h + self.y
            rotated_corners.append((rx, ry))

        return rotated_corners

    def get_edges(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return edges of the car as line segments for collision detection."""
        corners = self.get_corners()
        return [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]

    def collides_with(self, other: "Car") -> bool:
        """
        Check collision with another car using Separating Axis Theorem (SAT)
        for two convex polygons (rectangles).
        """
        return check_rect_collision(
            self.get_corners(), other.get_corners()
        )


def check_rect_collision(
    rect1: list[tuple[float, float]],
    rect2: list[tuple[float, float]],
) -> bool:
    """
    Check if two convex polygons (4-corner rectangles) overlap using SAT.
    Returns True if they collide.
    """
    for rect in [rect1, rect2]:
        for i in range(len(rect)):
            p1 = rect[i]
            p2 = rect[(i + 1) % len(rect)]

            # Edge direction
            edge_x = p2[0] - p1[0]
            edge_y = p2[1] - p1[1]

            # Perpendicular axis for projection
            axis_x = -edge_y
            axis_y = edge_x

            # Normalize axis
            length = math.hypot(axis_x, axis_y)
            if length < 1e-6:
                continue
            axis_x /= length
            axis_y /= length

            # Project rect1
            min1 = float("inf")
            max1 = float("-inf")
            for corner in rect1:
                proj = corner[0] * axis_x + corner[1] * axis_y
                min1 = min(min1, proj)
                max1 = max(max1, proj)

            # Project rect2
            min2 = float("inf")
            max2 = float("-inf")
            for corner in rect2:
                proj = corner[0] * axis_x + corner[1] * axis_y
                min2 = min(min2, proj)
                max2 = max(max2, proj)

            # Check for gap
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found, no collision

    return True  # No separating axis found, collision
