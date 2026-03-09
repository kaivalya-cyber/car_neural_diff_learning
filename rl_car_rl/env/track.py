import numpy as np
import math

class Track:
    def __init__(self, track_width=80):
        self.track_width = track_width
        self.center_points = []
        self.outer_boundary = []
        self.inner_boundary = []
        self.start_pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        
    def generate(self, track_width=80.0, num_control_points=12, max_radius=400, min_radius=150):
        """Generates a procedural closed-loop track using random angles and radii."""
        self.track_width = track_width
        
        # Generate random radii for control points
        angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)
        radii = np.random.uniform(min_radius, max_radius, size=num_control_points)
        
        # Center of the track (arbitrary anchor to avoid negatives initially)
        cx, cy = 500.0, 500.0
        
        # Calculate control points
        control_points = []
        for angle, radius in zip(angles, radii):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            control_points.append((x, y))
            
        # Catmull-Rom Spline interpolation to make smooth track
        self.center_points = self._compute_catmull_rom_spline(control_points, points_per_segment=10)
        
        # Generate inner and outer boundaries
        self._generate_boundaries()
        
        # Pick random starting position
        self._pick_start_pose()

    def _compute_catmull_rom_spline(self, control_points, points_per_segment=10):
        """Interpolates control points using a Catmull-Rom spline for a closed loop."""
        spline_points = []
        n = len(control_points)
        for i in range(n):
            p0 = control_points[(i - 1) % n]
            p1 = control_points[i]
            p2 = control_points[(i + 1) % n]
            p3 = control_points[(i + 2) % n]
            
            for t in np.linspace(0, 1, points_per_segment, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                
                # Catmull-Rom weights
                f1 = -0.5 * t3 + t2 - 0.5 * t
                f2 = 1.5 * t3 - 2.5 * t2 + 1.0
                f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                f4 = 0.5 * t3 - 0.5 * t2
                
                x = f1 * p0[0] + f2 * p1[0] + f3 * p2[0] + f4 * p3[0]
                y = f1 * p0[1] + f2 * p1[1] + f3 * p2[1] + f4 * p3[1]
                spline_points.append((x, y))
                
        return spline_points

    def _generate_boundaries(self):
        """Generates outer and inner boundary using normal vectors to the spline."""
        self.outer_boundary = []
        self.inner_boundary = []
        n = len(self.center_points)
        
        for i in range(n):
            p1 = self.center_points[i]
            p2 = self.center_points[(i + 1) % n]
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
                
            # Normal vector (rotate 90 degrees)
            nx = -dy / length
            ny = dx / length
            
            # Offset by half track width
            half_width = self.track_width / 2.0
            
            # Inner (left) and outer (right) bound based on counter-clockwise direction
            # Normal (nx, ny) points inward.
            # So adding it goes to the inner boundary, subtracting goes to outer boundary.
            self.inner_boundary.append((p1[0] + nx * half_width, p1[1] + ny * half_width))
            self.outer_boundary.append((p1[0] - nx * half_width, p1[1] - ny * half_width))

    def _pick_start_pose(self):
        """Randomly selects a start point on the spline and aligns heading with tangent."""
        idx = np.random.randint(0, len(self.center_points))
        next_idx = (idx + 1) % len(self.center_points)
        
        p = self.center_points[idx]
        pn = self.center_points[next_idx]
        
        dx = pn[0] - p[0]
        dy = pn[1] - p[1]
        heading = math.atan2(dy, dx)
        
        self.start_pose = {
            "x": p[0],
            "y": p[1],
            "heading": heading
        }

    def get_boundaries(self):
        return self.outer_boundary, self.inner_boundary

    def check_collision(self, car_corners):
        """
        Since the track is procedurally generated as a closed loop polygon,
        the track bounds are complex polygons.
        Car is inside track if all corners are inside the outer boundary AND
        no corners are inside the inner boundary.
        Return True if collision happens.
        """
        for corner in car_corners:
            if not self._is_point_in_polygon(corner, self.outer_boundary) or \
               self._is_point_in_polygon(corner, self.inner_boundary):
                return True
        return False
        
    def _is_point_in_polygon(self, point, polygon):
        """Ray-casting algorithm for complex polygons."""
        x, y = point
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside
        
    def get_distance_along_track(self, x, y):
        """
        Finds the closest point on the center spline and returns its normalized index.
        """
        if not self.center_points:
            return 0.0
            
        points = np.array(self.center_points)
        target = np.array([x, y])
        
        diff = points - target
        dists = np.sum(diff**2, axis=1)
        closest_idx = np.argmin(dists)
        
        return float(closest_idx) / len(self.center_points)
