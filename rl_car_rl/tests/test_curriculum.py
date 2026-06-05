import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.curriculum import CurriculumManager
from env.environment import CarEnv

class TestCurriculum(unittest.TestCase):
    def test_curriculum_scaling(self):
        manager = CurriculumManager(start_level=0.0, increase_threshold=40.0, min_samples=1)
        
        # Test easy params at start
        easy_params = manager.get_generation_params()
        self.assertEqual(easy_params["track_width"], 120.0)
        self.assertEqual(easy_params["num_control_points"], 8)
        
        # Add rewards above threshold and trigger update
        for _ in range(5):
            manager.add_reward(50.0)
        new_lvl = manager.update()
        self.assertGreater(new_lvl, 0.0)
        
        harder_params = manager.get_generation_params()
        self.assertLess(harder_params["track_width"], 120.0)
        self.assertGreater(harder_params["max_radius"], 350.0)
    
    def test_curriculum_bidirectional(self):
        manager = CurriculumManager(
            start_level=0.5, increase_threshold=40.0,
            decrease_threshold=10.0, min_samples=1,
        )
        self.assertEqual(manager.level, 0.5)
        
        # Add low rewards to trigger decrease
        for _ in range(5):
            manager.add_reward(5.0)
        new_lvl = manager.update()
        self.assertLess(new_lvl, 0.5)

    def test_env_difficulty_setting(self):
        env = CarEnv()
        env.set_difficulty({"track_width": 45.0, "num_control_points": 10})
        env.reset()
        
        # ensure track width propagated to track generation
        self.assertEqual(env.track.track_width, 45.0)

if __name__ == '__main__':
    unittest.main()
