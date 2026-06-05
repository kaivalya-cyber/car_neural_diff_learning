import numpy as np
import math


class Track:
    TRACK_TYPES = ["procedural", "oval", "figure_8", "multi_loop"]

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
        self.track_type = "procedural"

    def generate(self, track_width=80.0, num_control_points=12, max_radius=400,
                 min_radius=150, obstacle_count=0, obstacle_radius=15.0,
                 track_type="procedural"):
        """Generates a track of the specified type."""
        self.track_width = track_width

        if track_type not in self.TRACK_TYPES:
            print(f"Warning: unknown track type '{track_type}', falling back to procedural")
            track_type = "procedural"

        self.track_type = track_type

        if track_type == "oval":
            self._generate_oval()
        elif track_type == "figure_8":
            self._generate_figure_8()
        elif track_type == "multi_loop":
            self._generate_multi_loop()
        else:
            self._generate_procedural(num_control_points, max_radius, min_radius)

        self._generate_boundaries()
        self._pick_start_pose()
        self.obstacles = self._generate_obstacles(obstacle_count, obstacle_radius)

    def _generate_procedural(self, num_control_points, max_radius, min_radius):
        """Generates a procedural closed-loop track using random angles and radii."""
        angles = np.linspace(0, 2 * np.pi, num_control_points, endpoint=False)
        radii = np.random.uniform(min_radius, max_radius, size=num_control_points)
        cx, cy = self.track_width_px / 2.0, self.track_height_px / 2.0
        control_points = []
        for angle, radius in zip(angles, radii):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            control_points.append((x, y))
        self.center_points = self._compute_catmull_rom_spline(
            control_points, points_per_segment=20
        )

    def _generate_oval(self):
        """Generate an oval track with two straightaways and two semicircles."""
        cx, cy = self.track_width_px / 2.0, self.track_height_px / 2.0
        straight_length = 500  # length of each straightaway
        curve_radius = 150     # radius of the semicircles

        points = []
        n_curve = 40   # points per semicircle
        n_straight = 40  # points per straightaway

        left_cx = cx - straight_length / 2.0
        right_cx = cx + straight_length / 2.0

        # Top straightaway (left to right)
        for i in range(n_straight):
            t = i / n_straight
            x = left_cx + t * straight_length
            y = cy - curve_radius
            points.append((x, y))

        # Right semicircle (clockwise from top to bottom)
        for i in range(n_curve):
            angle = -math.pi / 2.0 - math.pi * i / n_curve  # -π/2 to -3π/2
            x = right_cx + curve_radius * math.cos(angle)
            y = cy + curve_radius * math.sin(angle)
            points.append((x, y))

        # Bottom straightaway (right to left)
        for i in range(n_straight):
            t = i / n_straight
            x = right_cx - t * straight_length
            y = cy + curve_radius
            points.append((x, y))

        # Left semicircle (clockwise from bottom to top)
        for i in range(n_curve):
            angle = math.pi / 2.0 - math.pi * i / n_curve  # π/2 to -π/2
            x = left_cx + curve_radius * math.cos(angle)
            y = cy + curve_radius * math.sin(angle)
            points.append((x, y))

        self.center_points = points

    def _generate_figure_8(self):
        """Generate a figure-8 track using a parametric Lemniscate of Gerono."""
        cx, cy = self.track_width_px / 2.0, self.track_height_px / 2.0
        # Scale the figure-8 to fill most of the track area
        a = 300  # horizontal extent
        b = 300  # vertical extent

        n_points = 400
        points = []
        for i in range(n_points):
            t = 2.0 * math.pi * i / n_points
            # Lemniscate of Gerono: x = a*cos(t), y = b*sin(t)*cos(t)
            x = cx + a * math.cos(t)
            y = cy + b * math.sin(t) * math.cos(t)
            points.append((x, y))

        self.center_points = points

    def _generate_multi_loop(self):
        """Generate a complex multi-loop track using parametric equations
        with epicycloid-like patterns."""
        cx, cy = self.track_width_px / 2.0, self.track_height_px / 2.0
        # Use a combination of sinusoidal modulation on a base circle
        # This creates a wavy, multi-lobed track shape
        R = 280  # base radius
        n_lobes = 3  # number of lobes/petals
        modulation = 0.45  # how much the radius varies

        n_points = 500
        points = []
        for i in range(n_points):
            theta = 2.0 * math.pi * i / n_points
            # Varying radius with multiple frequency components
            r = R * (1.0 + modulation * math.sin(n_lobes * theta))
            # Add a secondary frequency for complexity
            r += R * 0.15 * math.sin(2 * n_lobes * theta + math.pi / 3)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.append((x, y))

        self.center_points = points

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

            v1 = np.array(
                [p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=np.float64
            )
            v2 = np.array(
                [p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=np.float64
            )

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

            half_width = self.track_width / 2.0
            self.inner_boundary.append(
                (p_curr[0] + nx * half_width, p_curr[1] + ny * half_width)
            )
            self.outer_boundary.append(
                (p_curr[0] - nx * half_width, p_curr[1] - ny * half_width)
            )

    def _pick_start_pose(self):
        """Randomly selects a start point on the spline and aligns heading with tangent."""
        idx = np.random.randint(0, len(self.center_points))
        next_idx = (idx + 1) % len(self.center_points)

        p = self.center_points[idx]
        pn = self.center_points[next_idx]

        dx = pn[0] - p[0]
        dy = pn[1] - p[1]
        heading = math.atan2(dy, dx)

        self.start_pose = {"x": p[0], "y": p[1], "heading": heading}

    def get_boundaries(self):
        return self.outer_boundary, self.inner_boundary

    def check_collision(self, car_corners):
        """Check if any car corner is outside track boundaries."""
        for corner in car_corners:
            if not self._is_point_in_polygon(
                corner, self.outer_boundary
            ) or self._is_point_in_polygon(corner, self.inner_boundary):
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

            intersect = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def get_distance_along_track(self, x, y):
        """Finds the closest point on the center spline and returns normalized index."""
        if not self.center_points:
            return 0.0

        points = np.array(self.center_points)
        target = np.array([x, y])
        diff = points - target
        dists = np.sum(diff**2, axis=1)
        closest_idx = np.argmin(dists)
        return float(closest_idx) / len(self.center_points)

    def get_center_distance(self, x, y):
        """Returns normalized distance [0, 1] from car to track centerline."""
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

    def _generate_obstacles(
        self, count: int, radius: float
    ) -> list[tuple[float, float, float]]:
        """Generate random circular obstacles placed on the track surface."""
        if count <= 0 or not self.center_points:
            return []

        obstacles = []
        n_center = len(self.center_points)
        half_width = self.track_width / 2.0
        attempts = 0
        max_attempts = count * 20
        min_spacing = radius * 4

        while len(obstacles) < count and attempts < max_attempts:
            attempts += 1
            idx = np.random.randint(0, n_center)
            cx, cy = self.center_points[idx]

            next_idx = (idx + 1) % n_center
            px, py = self.center_points[next_idx]
            dx = px - cx
            dy = py - cy
            track_dir_len = math.hypot(dx, dy)
            if track_dir_len < 1e-6:
                continue
            nx = -dy / track_dir_len
            ny = dx / track_dir_len

            offset = np.random.uniform(-half_width * 0.7, half_width * 0.7)
            ox = cx + nx * offset
            oy = cy + ny * offset

            too_close = False
            for ex, ey, er in obstacles:
                if math.hypot(ox - ex, oy - ey) < (radius + er + min_spacing):
                    too_close = True
                    break

            sx, sy = self.start_pose["x"], self.start_pose["y"]
            if math.hypot(ox - sx, oy - sy) < radius * 10:
                too_close = True

            if not too_close:
                obstacles.append((ox, oy, radius))

        return obstacles

    def get_obstacles(self) -> list[tuple[float, float, float]]:
        """Return list of obstacles as (x, y, radius) tuples."""
        return self.obstacles

    def check_obstacle_collision(
        self, car_corners: list[tuple[float, float]]
    ) -> bool:
        """Check if any car corner collides with any obstacle."""
        for ox, oy, radius in self.obstacles:
            for corner in car_corners:
                cx, cy = corner
                if math.hypot(cx - ox, cy - oy) < radius:
                    return True
        return False
