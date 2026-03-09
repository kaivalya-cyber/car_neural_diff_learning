import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.track import Track

class TestProceduralTrack(unittest.TestCase):
    def setUp(self):
        self.track = Track()

    def test_track_generation(self):
        self.track.generate()
        
        # Test boundary generation
        outer, inner = self.track.get_boundaries()
        self.assertTrue(len(outer) > 10)
        self.assertTrue(len(inner) > 10)
        
        # Test that start pose was created
        pose = self.track.start_pose
        self.assertIn("x", pose)
        self.assertIn("y", pose)
        self.assertIn("heading", pose)
        
        # Start pose should be inside the boundaries 
        # (It's on the centerline, so collision check with a point representing car should be False)
        # However, car check requires 4 corners. So let's create a tiny polygon.
        tiny_box = [
            (pose["x"]-1, pose["y"]-1),
            (pose["x"]+1, pose["y"]-1),
            (pose["x"]+1, pose["y"]+1),
            (pose["x"]-1, pose["y"]+1)
        ]
        
        # Since it's on centerline, it shouldn't collide
        self.assertFalse(self.track.check_collision(tiny_box))

if __name__ == '__main__':
    unittest.main()
