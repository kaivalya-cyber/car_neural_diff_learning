import torch
import torch.nn as nn
import os


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm, Linear, and GELU activation."""

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class DriveNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ):
        """
        Residual MLP network for driving policy and value estimation.

        Args:
            input_size: Observation dimension.
            output_size: Action dimension (or 1 for value network).
            hidden_size: Width of hidden layers.
            num_blocks: Number of residual blocks.
            dropout: Dropout probability (0 = disabled).
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.output_head(x)

    def save_checkpoint(self, path: str) -> None:
        """Saves the model state dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Loads the model state dict."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        else:
            print(f"Warning: Checkpoint not found at {path}")
