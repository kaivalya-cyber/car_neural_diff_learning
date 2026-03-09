import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.curriculum import CurriculumManager
from env.environment import CarEnv

class TestCurriculum(unittest.TestCase):
    def test_curriculum_scaling(self):
        manager = CurriculumManager(start_level=0.0, threshold=40.0)
        
        # Test easy params at start
        easy_params = manager.get_generation_params()
        self.assertEqual(easy_params["track_width"], 120.0)
        self.assertEqual(easy_params["num_control_points"], 8)
        
        # Test level update
        new_lvl = manager.update(50.0) # passes threshold
        self.assertGreater(new_lvl, 0.0)
        
        harder_params = manager.get_generation_params()
        self.assertLess(harder_params["track_width"], 120.0) # width gets smaller
        self.assertGreater(harder_params["max_radius"], 350.0) # radius grows

    def test_env_difficulty_setting(self):
        env = CarEnv()
        env.set_difficulty({"track_width": 45.0, "num_control_points": 10})
        env.reset()
        
        # ensure track width propagated to track generation
        self.assertEqual(env.track.track_width, 45.0)

if __name__ == '__main__':
    unittest.main()
