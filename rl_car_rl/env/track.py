import numpy as np
import math

class Track:
    def __init__(self, track_width=80):
        # track_width is the drivable lane width, not the world size
        self.track_width = track_width
        # World size used by the renderer
        self.track_width_px = 1000
        self.track_height_px = 1000
        self.center_points = []
        self.outer_boundary = []
        self.inner_boundary = []
        self.start_pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.obstacles = []  # list of (x, y, radius) tuples
        
    def generate(self, track_width=80.0, num_control_points=12, max_radius=400, min_radius=150, obstacle_count=0, obstacle_radius=15.0):
        """Generates a procedural closed-loop track using random angles and radii."""
        self.track_width = track_width
        
        # Generate random radii for control points
        angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)
        radii = np.random.uniform(min_radius, max_radius, size=num_control_points)
        
        # Center of the track (anchor to world dimensions)
        cx, cy = self.track_width_px / 2.0, self.track_height_px / 2.0
        
        # Calculate control points
        control_points = []
        for angle, radius in zip(angles, radii):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            control_points.append((x, y))
            
        # Catmull-Rom Spline interpolation to make smooth track
        self.center_points = self._compute_catmull_rom_spline(control_points, points_per_segment=20)
        
        # Generate inner and outer boundaries
        self._generate_boundaries()
        
        # Pick random starting position
        self._pick_start_pose()
        
        # Generate obstacles if requested
        self.obstacles = self._generate_obstacles(obstacle_count, obstacle_radius)

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
            p_prev = self.center_points[(i - 1) % n]
            p_curr = self.center_points[i]
            p_next = self.center_points[(i + 1) % n]

            v1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=np.float64)
            v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=np.float64)

            len1 = math.hypot(v1[0], v1[1])
            len2 = math.hypot(v2[0], v2[1])
            if len1 == 0 or len2 == 0:
                continue

            v1 /= len1
            v2 /= len2

            # Use averaged tangent to reduce sharp boundary kinks
            tangent = v1 + v2
            tlen = math.hypot(tangent[0], tangent[1])
            if tlen < 1e-6:
                tangent = v2
                tlen = math.hypot(tangent[0], tangent[1])
                if tlen < 1e-6:
                    continue
            tangent /= tlen

            # Normal vector (rotate 90 degrees)
            nx = -tangent[1]
            ny = tangent[0]

            # Offset by half track width
            half_width = self.track_width / 2.0

            # Inner (left) and outer (right) bound based on counter-clockwise direction
            # Normal (nx, ny) points inward.
            self.inner_boundary.append((p_curr[0] + nx * half_width, p_curr[1] + ny * half_width))
            self.outer_boundary.append((p_curr[0] - nx * half_width, p_curr[1] - ny * half_width))

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
    
    def get_center_distance(self, x, y):
        """
        Returns the normalized distance [0, 1] from the car to the track centerline.
        0 = on centerline, 1 = at or beyond track edge.
        """
        if not self.center_points or not self.outer_boundary or not self.inner_boundary:
            return 0.0
        
        points = np.array(self.center_points)
        target = np.array([x, y])
        diff = points - target
        dists = np.sum(diff**2, axis=1)
        closest_idx = np.argmin(dists)
        
        closest_center = points[closest_idx]
        dist_to_center = float(np.linalg.norm(target - closest_center))
        
        half_width = self.track_width / 2.0
        normalized = min(dist_to_center / half_width, 1.0)
        return normalized

    def _generate_obstacles(self, count: int, radius: float) -> list[tuple[float, float, float]]:
        """
        Generate random circular obstacles placed on the track surface.
        Obstacles are placed at random positions along the centerline with
        a random offset within the track bounds.
        """
        if count <= 0 or not self.center_points:
            return []

        obstacles = []
        n_center = len(self.center_points)
        half_width = self.track_width / 2.0
        attempts = 0
        max_attempts = count * 20

        min_spacing = radius * 4  # minimum distance between obstacle centers

        while len(obstacles) < count and attempts < max_attempts:
            attempts += 1

            # Pick a random point on the centerline
            idx = np.random.randint(0, n_center)
            cx, cy = self.center_points[idx]

            # Random offset perpendicular to track direction
            next_idx = (idx + 1) % n_center
            px, py = self.center_points[next_idx]
            dx = px - cx
            dy = py - cy
            track_dir_len = math.hypot(dx, dy)
            if track_dir_len < 1e-6:
                continue
            # Normal perpendicular
            nx = -dy / track_dir_len
            ny = dx / track_dir_len

            # Random offset within track bounds
            offset = np.random.uniform(-half_width * 0.7, half_width * 0.7)
            ox = cx + nx * offset
            oy = cy + ny * offset

            # Check this obstacle doesn't overlap with existing ones
            too_close = False
            for ex, ey, er in obstacles:
                if math.hypot(ox - ex, oy - ey) < (radius + er + min_spacing):
                    too_close = True
                    break

            # Check it's not too close to the start position
            sx, sy = self.start_pose["x"], self.start_pose["y"]
            if math.hypot(ox - sx, oy - sy) < radius * 10:
                too_close = True

            if not too_close:
                obstacles.append((ox, oy, radius))

        return obstacles

    def get_obstacles(self) -> list[tuple[float, float, float]]:
        """Return list of obstacles as (x, y, radius) tuples."""
        return self.obstacles

    def check_obstacle_collision(self, car_corners: list[tuple[float, float]]) -> bool:
        """
        Check if any car corner collides with any obstacle.
        Returns True if collision detected.
        """
        for ox, oy, radius in self.obstacles:
            for corner in car_corners:
                cx, cy = corner
                if math.hypot(cx - ox, cy - oy) < radius:
                    return True
        return False
