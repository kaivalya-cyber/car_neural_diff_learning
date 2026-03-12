import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.policy import RacingPolicy
from agent.neural_network import DriveNetwork
import os

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOTrainer:
    def __init__(self, state_dim=9, action_dim=2, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, device="cpu"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device(device)
        self.state_dim = state_dim
        
        # Policy network
        self.policy = RacingPolicy(state_dim, action_dim, device)
        self.optimizer = optim.Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': self.policy.log_std, 'lr': lr}
        ])
        
        # Value network
        self.value_net = DriveNetwork(state_dim, 1).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for Value Network!")
            self.value_net = nn.DataParallel(self.value_net)
            
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # old_states is a list of [num_envs, state_dim] -> stack to [T, num_envs, state_dim]
        # Then flatten to [T * num_envs, state_dim]
        T = len(memory.states)
        num_envs = memory.states[0].shape[0] if len(memory.states[0].shape) > 1 else 1
        
        old_states = torch.tensor(np.stack(memory.states), dtype=torch.float32, device=self.device).view(-1, self.state_dim)
        old_actions = torch.tensor(np.stack(memory.actions), dtype=torch.float32, device=self.device).view(-1, 2)
        old_logprobs = torch.tensor(np.stack(memory.logprobs), dtype=torch.float32, device=self.device).view(-1)
        
        rewards = np.zeros((T, num_envs), dtype=np.float32)
        rewards_list = []
        discounted_reward = np.zeros(num_envs, dtype=np.float32)
        
        # Calculate discounted rewards per environment
        for t in reversed(range(T)):
            # Memory contains [num_envs] arrays for rewards and terminals
            r = np.array(memory.rewards[t])
            is_term = np.array(memory.is_terminals[t])
            
            # If terminal, reset discounted reward for that env to 0
            discounted_reward = np.where(is_term, 0.0, discounted_reward)
            discounted_reward = r + (self.gamma * discounted_reward)
            rewards[t] = discounted_reward
            
        # Flatten rewards natively onto GPU
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_state_values = self.value_net(old_states).squeeze(-1).detach()
        advantages = rewards - old_state_values
        
        # Tracking metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Optimize policy and value networks
        for _ in range(self.K_epochs):
            means = self.policy.net(old_states)
            stds = self.policy.log_std.exp().expand_as(means)
            dist = torch.distributions.Normal(means, stds)
            
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.value_net(old_states).squeeze(-1)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards).mean()
            
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
        return total_policy_loss / self.K_epochs, total_value_loss / self.K_epochs
            
    def save(self, is_best=False, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = f"{checkpoint_dir}/latest.pth" if not is_best else f"{checkpoint_dir}/best.pth"
        
        policy_sd = self.policy.net.module.state_dict() if isinstance(self.policy.net, nn.DataParallel) else self.policy.net.state_dict()
        value_sd = self.value_net.module.state_dict() if isinstance(self.value_net, nn.DataParallel) else self.value_net.state_dict()
        
        torch.save({
            'policy_net': policy_sd,
            'policy_log_std': self.policy.log_std,
            'policy_optimizer': self.optimizer.state_dict(),
            'value_net': value_sd,
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)

    def load(self, path):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
            
        checkpoint = torch.load(path)
        if isinstance(self.policy.net, nn.DataParallel):
            self.policy.net.module.load_state_dict(checkpoint['policy_net'])
        else:
            self.policy.net.load_state_dict(checkpoint['policy_net'])
            
        self.policy.log_std.data = checkpoint['policy_log_std']
        self.optimizer.load_state_dict(checkpoint['policy_optimizer'])
        
        if isinstance(self.value_net, nn.DataParallel):
            self.value_net.module.load_state_dict(checkpoint['value_net'])
        else:
            self.value_net.load_state_dict(checkpoint['value_net'])
            
        return True
