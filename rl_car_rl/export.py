"""
Model export utilities for deploying trained policies without the full training stack.

Usage:
    python export.py --checkpoint checkpoints/best.pth --output exported/model.pt
"""

import argparse
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.neural_network import DriveNetwork


class ExportedPolicy(torch.nn.Module):
    """Standalone policy wrapper for TorchScript export."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = DriveNetwork(state_dim, action_dim, hidden_size=hidden_size)
        self.log_std = torch.nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic forward pass returning (steering, throttle)."""
        mean = self.net(x)
        # Clamp steering [-1, 1], throttle [0, 1]
        steering = torch.clamp(mean[:, 0:1], -1.0, 1.0)
        throttle = torch.clamp(mean[:, 1:2], 0.0, 1.0)
        return steering, throttle


class InferenceRunner:
    """Lightweight inference runner for exported models."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Run inference on a single observation.

        Args:
            observation: numpy array of shape (state_dim,) or (batch, state_dim).

        Returns:
            numpy array of shape (2,) or (batch, 2) with [steering, throttle].
        """
        single_input = len(observation.shape) == 1
        if single_input:
            observation = np.expand_dims(observation, 0)

        x = torch.tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            steering, throttle = self.model(x)

        actions = torch.cat([steering, throttle], dim=-1).cpu().numpy()

        if single_input:
            actions = actions[0]
        return actions


def export_model(checkpoint_path: str, output_path: str, state_dim: int = 20) -> str:
    """
    Export a trained policy to TorchScript.

    Args:
        checkpoint_path: Path to the training checkpoint (.pth).
        output_path: Path to save the exported TorchScript model.
        state_dim: Observation dimension.

    Returns:
        Path to the exported model.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Determine action_dim from log_std shape
    log_std = checkpoint.get("policy_log_std")
    if log_std is not None:
        action_dim = log_std.shape[-1]
    else:
        action_dim = 2

    # Determine hidden_size from policy_net weights
    policy_weights = checkpoint.get("policy_net", {})
    hidden_size = 256
    for key in policy_weights:
        if "weight" in key:
            hidden_size = policy_weights[key].shape[0]
            break

    # Create model and load weights
    model = ExportedPolicy(state_dim, action_dim, hidden_size=hidden_size)
    model.net.load_state_dict(policy_weights)
    if log_std is not None:
        model.log_std.data = log_std

    model.eval()

    # Trace with example input
    example_input = torch.randn(1, state_dim)
    traced = torch.jit.trace(model, example_input)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    traced.save(output_path)
    print(f"Model exported to {output_path}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}, Hidden size: {hidden_size}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        test_out = loaded(example_input)
    print(f"  Verified: output shapes = {[o.shape for o in test_out]}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained policy to TorchScript")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pth",
        help="Path to training checkpoint",
    )
    parser.add_argument(
        "--output",
        default="exported/model.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=20,
        help="Observation dimension (sensor_count + 4)",
    )
    args = parser.parse_args()

    export_model(args.checkpoint, args.output, args.state_dim)
