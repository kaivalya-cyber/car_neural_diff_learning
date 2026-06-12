import os
import torch
import torch.nn as nn
import glob
import re


def discover_checkpoints(pattern: str = "checkpoints/best_ep*.pth") -> list[str]:
    return sorted(glob.glob(pattern),
                  key=lambda x: int(re.search(r'ep(\d+)', x).group(1)) if re.search(r'ep(\d+)', x) else 0)


def average_checkpoints(checkpoint_paths: list[str], device: str = "cpu") -> dict:
    if not checkpoint_paths:
        raise ValueError("No checkpoints to average")

    avg_state = None
    num = len(checkpoint_paths)
    print(f"Averaging {num} checkpoints...")

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        if avg_state is None:
            avg_state = {k: ckpt["policy_net"][k].float() / num for k in ckpt["policy_net"]}
        else:
            for k in avg_state:
                avg_state[k] += ckpt["policy_net"][k].float() / num

    return avg_state


def create_ensemble_model(checkpoint_paths: list[str], state_dim: int, action_dim: int = 2,
                          hidden_size: int = 256, num_blocks: int = 2, device: str = "cpu"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.policy import RacingPolicy

    models = []
    for path in checkpoint_paths:
        policy = RacingPolicy(state_dim, action_dim, device, hidden_size=hidden_size, num_blocks=num_blocks)
        ckpt = torch.load(path, map_location=device)
        if isinstance(policy.net, nn.DataParallel):
            policy.net.module.load_state_dict(ckpt["policy_net"])
        else:
            policy.net.load_state_dict(ckpt["policy_net"])
        policy.eval()
        models.append(policy)

    class Ensemble:
        def __init__(self, models):
            self.models = models

        @torch.no_grad()
        def act(self, state, deterministic=True):
            actions = []
            for m in self.models:
                action, _, _, _ = m.get_action(state, deterministic=deterministic)
                actions.append(action)
            return torch.stack(actions).mean(dim=0)

    return Ensemble(models)


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    paths = discover_checkpoints()
    if len(paths) < 2:
        print(f"Need at least 2 checkpoints, found {len(paths)}")
        return

    print(f"Found {len(paths)} checkpoints:")
    for p in paths:
        print(f"  {p}")

    avg_state = average_checkpoints(paths)
    print(f"Created averaged checkpoint with {len(avg_state)} keys")

    output_path = "checkpoints/ensemble.pth"
    torch.save({"policy_net": avg_state}, output_path)
    print(f"Saved ensemble to {output_path}")


if __name__ == "__main__":
    import sys
    main()
