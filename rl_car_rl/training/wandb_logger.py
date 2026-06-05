"""
Weights & Biases (wandb) integration for cloud experiment tracking.

Usage:
    from training.wandb_logger import WandbLogger
    logger = WandbLogger(config=config, project="rl-car")
    logger.log({"reward": 42.0, "loss": 0.1}, step=100)
    logger.close()
"""

import os
import time
from typing import Optional


class WandbLogger:
    """Lightweight wandb wrapper with graceful fallback when not installed."""

    def __init__(self, config: dict, project: str = "rl-car-racing",
                 name: Optional[str] = None, tags: Optional[list] = None,
                 notes: Optional[str] = None):
        self.enabled = False
        self.run = None
        try:
            import wandb
            self.run = wandb.init(
                project=project,
                config=config,
                name=name,
                tags=tags or [],
                notes=notes,
                reinit=True,
            )
            self.enabled = True
            print(f"WandB logging enabled: {wandb.run.url}")
        except ImportError:
            print("WandB not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"WandB init failed: {e}")

    def log(self, metrics: dict, step: Optional[int] = None, commit: bool = True):
        if self.enabled and self.run:
            import wandb
            wandb.log(metrics, step=step, commit=commit)

    def log_scalar(self, name: str, value: float, step: int):
        self.log({name: value}, step=step)

    def log_metrics(self, metrics: dict, step: int):
        self.log(metrics, step=step)

    def summary(self, key: str, value):
        if self.enabled and self.run:
            self.run.summary[key] = value

    def save_checkpoint(self, path: str):
        if self.enabled and self.run:
            import wandb
            if os.path.exists(path):
                wandb.save(path, policy="now")

    def close(self, exit_code: int = 0):
        if self.enabled and self.run:
            import wandb
            try:
                wandb.finish(exit_code=exit_code)
            except TypeError:
                wandb.finish()  # older wandb versions don't accept exit_code
            self.enabled = False
