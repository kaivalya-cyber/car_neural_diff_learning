import torch
import numpy as np
from agent.neural_network import DriveNetwork
from torch.distributions import Normal


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.2,
        theta: float = 0.15,
        dt: float = 1e-2,
        sigma_decay: float = 1.0,
    ):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.sigma_decay = sigma_decay
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Sample noise and decay sigma."""
        dx = self.theta * (-self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = self.state + dx
        # Decay sigma
        self.sigma *= self.sigma_decay
        return self.state.astype(np.float32)


class RacingPolicy:
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 2,
        device: str = "cpu",
        ou_sigma: float = 0.0,
        ou_theta: float = 0.15,
        ou_sigma_decay: float = 1.0,
    ):
        self.device = torch.device(device)
        self.net = DriveNetwork(state_dim, action_dim).to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for Policy Network!")
            self.net = torch.nn.DataParallel(self.net)

        self.log_std = torch.nn.Parameter(
            torch.zeros(1, action_dim, device=self.device)
        )
        self.action_dim = action_dim

        # OU noise for exploration
        self.ou_noise = None
        if ou_sigma > 0:
            self.ou_noise = OrnsteinUhlenbeckNoise(
                action_dim=action_dim,
                sigma=ou_sigma,
                theta=ou_theta,
                sigma_decay=ou_sigma_decay,
            )

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: numpy array of shape (num_envs, state_dim) or (state_dim,).
            deterministic: If True, return mean action without sampling.

        Returns:
            final_action: numpy array of clamped actions.
            raw_action: tensor of unclamped sampled actions.
            log_prob: tensor of log probabilities.
            mean: tensor of action means.
        """
        state = np.array(state)
        single_input = len(state.shape) == 1
        if single_input:
            state = np.expand_dims(state, 0)

        num_envs = state.shape[0]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad() if deterministic else torch.enable_grad():
            mean = self.net(state_tensor).to(self.device)
            std = self.log_std.exp().expand_as(mean).to(self.device)
            dist = Normal(mean, std)

            if deterministic:
                action = mean
            else:
                action = dist.sample().to(self.device)

            # Add OU noise for exploration (on top of stochastic policy)
            if self.ou_noise is not None and not deterministic:
                ou = torch.tensor(
                    np.stack([self.ou_noise.sample() for _ in range(num_envs)]),
                    dtype=torch.float32,
                    device=self.device,
                )
                action = action + ou

            log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.detach().cpu().numpy()

            # Action clamping
            final_action = np.zeros_like(action_np)
            final_action[:, 0] = np.clip(action_np[:, 0], -1.0, 1.0)  # steering
            final_action[:, 1] = np.clip(action_np[:, 1], 0.0, 1.0)    # throttle

        if single_input:
            final_action = final_action[0]
            action = action[0]
            log_prob = log_prob[0]
            mean = mean[0]

        return final_action, action, log_prob, mean

    def reset_noise(self) -> None:
        """Reset OU noise state (call at start of each episode)."""
        if self.ou_noise is not None:
            self.ou_noise.reset()
