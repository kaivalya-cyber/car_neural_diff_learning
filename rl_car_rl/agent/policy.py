import torch
import numpy as np
from agent.neural_network import DriveNetwork
from torch.distributions import Normal

class RacingPolicy:
    def __init__(self, state_dim=9, action_dim=2, device="cpu"):
        self.device = torch.device(device)
        self.net = DriveNetwork(state_dim, action_dim).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for Policy Network!")
            self.net = torch.nn.DataParallel(self.net)
            
        self.log_std = torch.nn.Parameter(torch.zeros(1, action_dim, device=self.device))
        
    def get_action(self, state, deterministic=False):
        """
        state: numpy array of shape (num_envs, state_dim) or (state_dim,)
        Returns batched numpy array of actions (num_envs, action_dim).
        """
        state = np.array(state)
        # Ensure 2D tensor
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
            
        # Push to device natively
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        with torch.no_grad() if deterministic else torch.enable_grad():
            # Ensure all tensors used by the distribution are on the same device
            mean = self.net(state_tensor).to(self.device)
            std = self.log_std.exp().expand_as(mean).to(self.device)
            dist = Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample().to(self.device)
                
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            action_np = action.detach().cpu().numpy()
            
            # Action clamping limits across batch
            final_action = np.zeros_like(action_np)
            final_action[:, 0] = np.clip(action_np[:, 0], -1.0, 1.0) # steering
            final_action[:, 1] = np.clip(action_np[:, 1], 0.0, 1.0) # throttle

        # If it was a single state input, squeeze back
        if len(state) == 1:
            final_action = final_action[0]
            action = action[0]
            log_prob = log_prob[0]
            mean = mean[0]

        return final_action, action, log_prob, mean
