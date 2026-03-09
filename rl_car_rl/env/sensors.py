import math
import numpy as np
from env.car import Car
from env.track import Track

class SensorSystem:
    def __init__(self, num_sensors=5, max_distance=150.0):
        self.num_sensors = num_sensors
        self.max_distance = max_distance
        
        # Ray angles relative to car heading: e.g., -90, -45, 0, 45, 90 for 5 sensors
        if num_sensors == 1:
            self.angles = [0.0]
        else:
            spread = math.pi  # 180 degrees spread
            self.angles = np.linspace(-spread/2, spread/2, num_sensors).tolist()

    def get_readings(self, car: Car, track: Track):
        """
        Returns normalized distance readings [0, 1] for each sensor.
        1 means no obstacle within max_distance.
        0 means obstacle is touching the sensor.
        """
        readings = []
        car_x = car.x
        car_y = car.y
        heading = car.heading
        
        outer_boundary, inner_boundary = track.get_boundaries()
        boundaries = [outer_boundary, inner_boundary]

        for angle in self.angles:
            ray_heading = heading + angle
            ray_dx = math.cos(ray_heading)
            ray_dy = math.sin(ray_heading)
            
            # Simple raycasting using step-wise sampling (can be optimized with line intersection)
            min_dist = self.max_distance
            
            # More precise line intersection
            p1 = (car_x, car_y)
            p2 = (car_x + ray_dx * self.max_distance, car_y + ray_dy * self.max_distance)
            
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
            
            # Normalize reading
            normalized_dist = min_dist / self.max_distance
            readings.append(normalized_dist)
            
        return readings

    def _line_intersection_distance(self, p1, p2, p3, p4):
        """
        Returns the distance from p1 to the intersection point of line segments p1-p2 and p3-p4.
        Returns None if no intersection.
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
