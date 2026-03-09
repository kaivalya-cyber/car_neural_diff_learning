import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import CarEnv

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = CarEnv()

    def test_environment_reset(self):
        # Validate environment reset()
        obs = self.env.reset()
        self.assertEqual(len(obs), 9)
        self.assertEqual(self.env.current_step, 0)
        self.assertIsInstance(obs, np.ndarray)

    def test_sensor_outputs(self):
        # Validate sensor outputs format and bounds
        obs = self.env.reset()
        sensor_readings = obs[:5]
        for reading in sensor_readings:
            self.assertTrue(0.0 <= reading <= 1.0)
            
    def test_step_execution(self):
        self.env.reset()
        action = [0.0, 1.0] # straight, full throttle
        obs, reward, done, info = self.env.step(action)
        self.assertEqual(len(obs), 9)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("crashed", info)

if __name__ == '__main__':
    unittest.main()
