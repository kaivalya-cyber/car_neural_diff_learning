import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.vector_env import VectorEnv

class TestVectorEnvironment(unittest.TestCase):
    def setUp(self):
        self.num_envs = 4
        self.sensor_count = 16
        self.state_dim = self.sensor_count + 4
        self.env = VectorEnv(num_envs=self.num_envs, sensor_count=self.sensor_count)

    def tearDown(self):
        self.env.close()

    def test_vector_reset(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (self.num_envs, self.state_dim))
        self.assertIsInstance(obs, np.ndarray)

    def test_vector_step(self):
        self.env.reset()
        actions = np.zeros((self.num_envs, 2))
        
        # Test step function returns batched components
        obs, rewards, dones, infos = self.env.step(actions)
        
        self.assertEqual(obs.shape, (self.num_envs, self.state_dim))
        self.assertEqual(rewards.shape, (self.num_envs,))
        self.assertEqual(dones.shape, (self.num_envs,))
        self.assertEqual(len(infos), self.num_envs)
        
        # Basic bounds verify
        self.assertTrue(all(isinstance(done, (bool, np.bool_)) for done in dones))
        self.assertTrue(all(isinstance(reward, np.floating) or isinstance(reward, float) for reward in rewards))

if __name__ == '__main__':
    unittest.main()
