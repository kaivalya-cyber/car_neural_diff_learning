import math
import numpy as np
from env.car import Car
from env.track import Track


class SensorSystem:
    def __init__(
        self,
        sensor_count: int = 16,
        max_distance: float = 150.0,
        spread_degrees: float = 360.0,
        front_density: float = 1.0,
    ):
        """
        LiDAR-style raycast sensor system.

        Args:
            sensor_count: Number of rays to cast.
            max_distance: Maximum detection distance in world units.
            spread_degrees: Angular spread of the sensor array in degrees.
                360 = full circle (LiDAR), 180 = front-facing only.
            front_density: Concentration of sensors toward the front.
                1.0 = uniform spacing, 2.0 = 2x denser in front, 0.5 = denser in sides.
        """
        self.sensor_count = sensor_count
        self.max_distance = max_distance
        self.spread_degrees = spread_degrees
        self.front_density = front_density

        # Generate ray angles with configurable front bias
        spread_rad = math.radians(spread_degrees)
        if sensor_count == 1:
            self.angles = [0.0]
        else:
            if abs(front_density - 1.0) < 1e-6:
                # Uniform spacing (fast path)
                self.angles = np.linspace(
                    -spread_rad / 2, spread_rad / 2, sensor_count
                ).tolist()
            else:
                # Non-uniform: map through a transformation that concentrates
                # angles near the front (angle=0)
                uniform = np.linspace(0, 1, sensor_count)
                # Apply power transform: values near 0.5 (center) get closer to 0.5
                # when front_density > 1, making front angles denser
                centered = (uniform - 0.5) * 2.0  # [-1, 1]
                mapped = np.sign(centered) * np.abs(centered) ** (1.0 / front_density)
                mapped = (mapped / 2.0 + 0.5)  # back to [0, 1]
                self.angles = (mapped - 0.5) * spread_rad
                self.angles = self.angles.tolist()

    def get_readings(self, car: Car, track: Track) -> list[float]:
        """
        Returns normalized distance readings [0, 1] for each sensor.
        1 means no obstacle within max_distance.
        0 means obstacle is touching the sensor.
        Detects both track boundaries and dynamic obstacles.
        """
        readings = []
        car_x = car.x
        car_y = car.y
        heading = car.heading

        outer_boundary, inner_boundary = track.get_boundaries()
        boundaries = [outer_boundary, inner_boundary]
        obstacles = track.get_obstacles()

        for angle in self.angles:
            ray_heading = heading + angle
            ray_dx = math.cos(ray_heading)
            ray_dy = math.sin(ray_heading)

            min_dist = self.max_distance

            # Check intersection with track boundaries
            p1 = (car_x, car_y)
            p2 = (
                car_x + ray_dx * self.max_distance,
                car_y + ray_dy * self.max_distance,
            )

            for boundary in boundaries:
                n = len(boundary)
                j = n - 1
                for i in range(n):
                    p3 = boundary[j]
                    p4 = boundary[i]

                    dist = self._line_intersection_distance(p1, p2, p3, p4)
                    if dist is not None and dist < min_dist:
                        min_dist = dist

                    j = i

            # Check intersection with obstacles (circle-ray intersection)
            for ox, oy, radius in obstacles:
                dist = self._ray_circle_intersection(
                    car_x, car_y, ray_dx, ray_dy, ox, oy, radius
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Normalize reading
            normalized_dist = min_dist / self.max_distance
            readings.append(normalized_dist)

        return readings

    def _ray_circle_intersection(
        self,
        rx: float,
        ry: float,
        rdx: float,
        rdy: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> float | None:
        """
        Ray-circle intersection. Returns distance to nearest intersection
        point along the ray, or None if no intersection.
        """
        # Vector from ray origin to circle center
        ocx = rx - cx
        ocy = ry - cy

        # Quadratic: (rdx^2 + rdy^2)*t^2 + 2*(ocx*rdx + ocy*rdy)*t + (ocx^2 + ocy^2 - r^2) = 0
        a = rdx * rdx + rdy * rdy  # should be 1.0 for normalized direction
        b = 2.0 * (ocx * rdx + ocy * rdy)
        c = ocx * ocx + ocy * ocy - radius * radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None

        # Nearest positive t
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        return None

    def _line_intersection_distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> float | None:
        """
        Returns the distance from p1 to the intersection point of
        line segments p1-p2 and p3-p4. Returns None if no intersection.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None  # Parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            dist = math.hypot(ix - x1, iy - y1)
            return dist

        return None
