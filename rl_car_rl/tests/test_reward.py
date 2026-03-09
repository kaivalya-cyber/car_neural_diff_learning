import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.reward_function import compute_reward

class TestReward(unittest.TestCase):
    def test_reward_forward_motion(self):
        reward = compute_reward(velocity=10.0, progress_diff=0.0, crashed=False, done=False)
        self.assertGreater(reward, -0.01) # positive from velocity, slight time penalty
        self.assertIsInstance(reward, float)

    def test_reward_crash_penalty(self):
        reward = compute_reward(velocity=10.0, progress_diff=0.0, crashed=True, done=True)
        self.assertLess(reward, -9.0) # Massive penalty

if __name__ == '__main__':
    unittest.main()
