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
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
        max_grad_norm: float = 0.5,
        lr_decay: float = 0.999,
        lr_warmup_epochs: int = 0,
        lr_warmup_start_factor: float = 0.1,
        entropy_coef: float = 0.01,
        entropy_decay: float = 1.0,
        normalize_rewards: bool = True,
        ou_sigma: float = 0.0,
        ou_theta: float = 0.15,
        ou_sigma_decay: float = 1.0,
        hidden_size: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_start_factor = lr_warmup_start_factor
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._update_count = 0
        self.initial_policy_lr = lr
        self.initial_value_lr = lr
        self.normalize_rewards = normalize_rewards

        # Policy network with optional OU noise
        self.policy = RacingPolicy(
            state_dim, action_dim, device,
            ou_sigma=ou_sigma,
            ou_theta=ou_theta,
            ou_sigma_decay=ou_sigma_decay,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.net.parameters(), "lr": lr},
                {"params": self.policy.log_std, "lr": lr},
            ]
        )

        # Value network
        self.value_net = DriveNetwork(
            state_dim, 1,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            dropout=dropout,
        ).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for Value Network!")
            self.value_net = nn.DataParallel(self.value_net)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )
        self.value_scheduler = optim.lr_scheduler.ExponentialLR(
            self.value_optimizer, gamma=lr_decay
        )

        self.MseLoss = nn.MSELoss()

    def update(self, memory: Memory, final_value: torch.Tensor | None = None) -> tuple[float, float]:
        """
        Proximal Policy Optimization update with GAE advantage estimation,
        gradient clipping, and entropy bonus.

        Args:
            memory: Collected trajectory memory.
            final_value: Value of the state after the last stored transition
                (shape: [num_envs]). Used to bootstrap GAE at trajectory boundaries.
                If None, zeros are used.
        """
        T = len(memory.states)
        num_envs = (
            memory.states[0].shape[0] if len(memory.states[0].shape) > 1 else 1
        )

        # Stack and flatten all trajectory data
        old_states = (
            torch.tensor(np.stack(memory.states), dtype=torch.float32, device=self.device)
            .view(-1, self.state_dim)
        )
        old_actions = (
            torch.tensor(np.stack(memory.actions), dtype=torch.float32, device=self.device)
            .view(-1, self.action_dim)
        )
        old_logprobs = (
            torch.tensor(np.stack(memory.logprobs), dtype=torch.float32, device=self.device)
            .view(-1)
        )

        # Compute state values for all stored states
        with torch.no_grad():
            values = self.value_net(old_states).squeeze(-1)  # [T * num_envs]
        values = values.view(T, num_envs)  # [T, num_envs]

        rewards_tensor = torch.tensor(
            np.stack(memory.rewards), dtype=torch.float32, device=self.device
        )  # [T, num_envs]
        terminals_tensor = torch.tensor(
            np.stack(memory.is_terminals), dtype=torch.float32, device=self.device
        )  # [T, num_envs]

        # --- Generalized Advantage Estimation (GAE) ---
        advantages = torch.zeros((T, num_envs), dtype=torch.float32, device=self.device)
        gae = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            # Value of next state
            if t == T - 1:
                # Bootstrap from final_value if provided, else zero
                if final_value is not None:
                    next_value = final_value.to(dtype=torch.float32, device=self.device)
                else:
                    next_value = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
            else:
                next_value = values[t + 1]

            mask = 1.0 - terminals_tensor[t]  # 0 if terminal, 1 otherwise

            delta = rewards_tensor[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = advantages + values  # [T, num_envs]

        # Flatten to [T * num_envs]
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # Normalize rewards per batch (before GAE if enabled)
        if self.normalize_rewards:
            r_mean = rewards_tensor.mean()
            r_std = rewards_tensor.std() + 1e-7
            rewards_tensor = (rewards_tensor - r_mean) / r_std

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # --- PPO Policy & Value Optimization ---
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.K_epochs):
            means = self.policy.net(old_states)
            stds = self.policy.log_std.exp().expand_as(means)
            dist = torch.distributions.Normal(means, stds)

            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.value_net(old_states).squeeze(-1)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                * advantages
            )

            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()
            value_loss = 0.5 * self.MseLoss(state_values, returns).mean()

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for training stability
            nn.utils.clip_grad_norm_(self.policy.net.parameters(), self.max_grad_norm)
            if isinstance(self.value_net, nn.DataParallel):
                nn.utils.clip_grad_norm_(
                    self.value_net.module.parameters(), self.max_grad_norm
                )
            else:
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.max_grad_norm
                )
            # Also clip log_std gradient
            nn.utils.clip_grad_norm_([self.policy.log_std], self.max_grad_norm)

            self.optimizer.step()
            self.value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        # LR warmup: linearly increase LR from start_factor*lr to full lr
        self._update_count += 1
        if self._update_count <= self.lr_warmup_epochs:
            warmup_frac = self._update_count / max(self.lr_warmup_epochs, 1)
            scale = self.lr_warmup_start_factor + (1.0 - self.lr_warmup_start_factor) * warmup_frac
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.initial_policy_lr * scale
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] = self.initial_value_lr * scale
        else:
            # Step the learning rate schedulers (only after warmup)
            self.policy_scheduler.step()
            self.value_scheduler.step()

        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay

        return total_policy_loss / self.K_epochs, total_value_loss / self.K_epochs

    def save(self, is_best: bool = False, checkpoint_dir: str = "checkpoints", episode: int | None = None) -> str:
        """Save model checkpoint. Returns the path saved to."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        if is_best and episode is not None:
            path = f"{checkpoint_dir}/best_ep{episode}.pth"
        elif is_best:
            path = f"{checkpoint_dir}/best.pth"
        else:
            path = f"{checkpoint_dir}/latest.pth"

        policy_sd = (
            self.policy.net.module.state_dict()
            if isinstance(self.policy.net, nn.DataParallel)
            else self.policy.net.state_dict()
        )
        value_sd = (
            self.value_net.module.state_dict()
            if isinstance(self.value_net, nn.DataParallel)
            else self.value_net.state_dict()
        )

        torch.save(
            {
                "policy_net": policy_sd,
                "policy_log_std": self.policy.log_std,
                "policy_optimizer": self.optimizer.state_dict(),
                "policy_scheduler": self.policy_scheduler.state_dict(),
                "value_net": value_sd,
                "value_optimizer": self.value_optimizer.state_dict(),
                "value_scheduler": self.value_scheduler.state_dict(),
                "entropy_coef": self.entropy_coef,
                "entropy_decay": self.entropy_decay,
                "state_dim": self.state_dim,
                "hidden_size": self.hidden_size,
                "num_blocks": self.num_blocks,
                "dropout": self.dropout,
            },
            path,
        )
        return path

    def load(self, path: str) -> bool:
        """Load model checkpoint. Returns True if successful."""
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.policy.net, nn.DataParallel):
            self.policy.net.module.load_state_dict(checkpoint["policy_net"])
        else:
            self.policy.net.load_state_dict(checkpoint["policy_net"])

        self.policy.log_std.data = checkpoint["policy_log_std"].to(self.device)
        self.optimizer.load_state_dict(checkpoint["policy_optimizer"])

        if "policy_scheduler" in checkpoint:
            self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler"])

        if isinstance(self.value_net, nn.DataParallel):
            self.value_net.module.load_state_dict(checkpoint["value_net"])
        else:
            self.value_net.load_state_dict(checkpoint["value_net"])

        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        if "value_scheduler" in checkpoint:
            self.value_scheduler.load_state_dict(checkpoint["value_scheduler"])

        if "entropy_coef" in checkpoint:
            self.entropy_coef = checkpoint["entropy_coef"]
        if "entropy_decay" in checkpoint:
            self.entropy_decay = checkpoint["entropy_decay"]

        return True
