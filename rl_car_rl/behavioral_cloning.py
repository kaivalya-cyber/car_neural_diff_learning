"""
Behavioral cloning: train a policy from a generated dataset,
then optionally fine-tune with PPO for a warm start.

Usage:
    python behavioral_cloning.py --dataset datasets/trajectories.npz --epochs 50
    python behavioral_cloning.py --dataset datasets/trajectories.npz --fine-tune --episodes 100
"""

import argparse
import os
import sys
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.policy import RacingPolicy
from training.trainer import PPOTrainer
from training.training_loop import train_agent


def clone_from_dataset(
    dataset_path: str,
    state_dim: int,
    action_dim: int,
    hidden_size: int = 256,
    num_blocks: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "auto",
    output_path: str = "checkpoints/cloned.pth",
) -> str:
    """
    Train a policy via behavioral cloning (supervised learning on actions).

    Returns path to the cloned checkpoint.
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Generate one first with: python generate_dataset.py")
        sys.exit(1)

    data = np.load(dataset_path)
    states = data["states"]
    actions = data["actions"]

    print(f"Cloning from {len(states)} samples, {state_dim}-dim states")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    # Filter out terminal transitions for cleaner cloning
    dones = data.get("dones", np.zeros(len(states), dtype=bool))
    mask = ~np.array(dones, dtype=bool)
    states = states[mask]
    actions = actions[mask]
    print(f"  After filtering terminals: {len(states)} samples")

    # Auto-detect device
    if device == "auto":
        import torch
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"  Device: {device}")

    # Create policy without OU noise
    policy = RacingPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
    )

    # Dataset
    dataset = TensorDataset(
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    device_obj = torch.device(device)

    policy.net.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device_obj)
            batch_actions = batch_actions.to(device_obj)

            predicted = policy.net(batch_states)
            loss = criterion(predicted, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

    # Save cloned checkpoint
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(
        {
            "policy_net": policy.net.state_dict(),
            "policy_log_std": policy.log_std.data,
            "state_dim": state_dim,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
        },
        output_path,
    )
    print(f"\nCloned model saved to {output_path}")
    print(f"  Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")

    return output_path


def fine_tune_with_ppo(
    cloned_path: str,
    episodes: int = 500,
    preset: str | None = None,
) -> None:
    """Fine-tune a cloned policy using PPO."""
    config_overrides = {}
    if preset:
        preset_path = os.path.join("configs", "presets", f"{preset}.yaml")
        if os.path.exists(preset_path):
            with open(preset_path, "r") as f:
                config_overrides = yaml.safe_load(f)

    # Set up training to resume from the cloned checkpoint
    config_overrides["max_episodes"] = episodes

    # Train with the cloned weights as starting point
    # We'll copy the cloned weights to the checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    import shutil
    
    # Warn if overwriting existing checkpoint
    target = "checkpoints/latest.pth"
    if os.path.exists(target):
        print(f"  Warning: overwriting existing {target}")
        bak = target + ".bak"
        shutil.copy(target, bak)
        print(f"  Backup saved to {bak}")
    shutil.copy(cloned_path, target)

    print(f"\nFine-tuning cloned policy with PPO for {episodes} episodes...")
    train_agent(
        config_overrides=config_overrides,
        resume=True,
        output_dir="experiments/clone_ft",
    )
    print("Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral cloning + PPO fine-tuning")
    parser.add_argument("--dataset", default="datasets/trajectories.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--output", default="checkpoints/cloned.pth")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--ft-episodes", type=int, default=500)
    parser.add_argument("--preset", type=str, default="")
    args = parser.parse_args()

    cloned = clone_from_dataset(
        dataset_path=args.dataset,
        state_dim=args.state_dim,
        action_dim=2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
    )

    if args.fine_tune:
        fine_tune_with_ppo(cloned, episodes=args.ft_episodes, preset=args.preset or None)
