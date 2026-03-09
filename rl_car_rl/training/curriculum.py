class CurriculumManager:
    def __init__(self, start_level=0.0, threshold=40.0, max_level=1.0, increase_rate=0.05):
        """
        Manages the difficulty level of track generation.
        Threshold: Average reward required to increase the level.
        """
        self.level = start_level
        self.threshold = threshold
        self.max_level = max_level
        self.increase_rate = increase_rate
        
        # Difficulty bounds for generation params
        # (width, min_radius, max_radius, num_control_points)
        self.easy_params = (120, 300, 350, 8)
        self.hard_params = (60, 100, 450, 16)
        
    def update(self, mean_reward):
        """Updates the current curriculum level based on recent performance."""
        # Simple threshold based scaling
        if mean_reward >= self.threshold:
            self.level = min(self.max_level, self.level + self.increase_rate)
            # We scale the threshold up so it gets progressively harder to advance
            self.threshold += 10.0
        
        # If performing very poorly, we could decay the level, but for now we only monotonic increase
        
        return self.level
        
    def get_generation_params(self):
        """Returns the interpolated track generation parameters for the current level."""
        # Linear interpolation
        ew, eminr, emaxr, encp = self.easy_params
        hw, hminr, hmaxr, hncp = self.hard_params
        
        w = ew + (hw - ew) * self.level
        minr = eminr + (hminr - eminr) * self.level
        maxr = emaxr + (hmaxr - emaxr) * self.level
        ncp = int(encp + (hncp - encp) * self.level)
        
        return {
            "track_width": w,
            "min_radius": minr,
            "max_radius": maxr,
            "num_control_points": ncp
        }
