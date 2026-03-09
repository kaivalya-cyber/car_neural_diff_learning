import torch
import torch.nn as nn
import os

class DriveNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DriveNetwork, self).__init__()
        
        # Dense 256, ReLU, Dense 256, ReLU, Dense 128, ReLU, Output Layer
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_size)
        Returns: logits/raw values
        """
        return self.network(x)

    def save_checkpoint(self, path: str):
        """Saves the model state dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads the model state dict."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        else:
            print(f"Warning: Checkpoint not found at {path}")
