import unittest
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.neural_network import DriveNetwork
from agent.policy import RacingPolicy

class TestNetwork(unittest.TestCase):
    def test_network_forward_pass(self):
        # Validate neural network forward pass
        net = DriveNetwork(input_size=9, output_size=2)
        dummy_input = torch.zeros((1, 9))
        output = net(dummy_input)
        self.assertEqual(output.shape, (1, 2))

    def test_policy_action_shape(self):
        policy = RacingPolicy(state_dim=9, action_dim=2)
        dummy_state = [0.5] * 9
        action, raw_action, log_prob, mean = policy.get_action(dummy_state)
        
        self.assertEqual(len(action), 2)
        self.assertTrue(-1.0 <= action[0] <= 1.0)
        self.assertTrue(0.0 <= action[1] <= 1.0)

if __name__ == '__main__':
    unittest.main()
