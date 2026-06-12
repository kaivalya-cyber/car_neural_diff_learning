import torch
import torch.nn as nn
import numpy as np


def magnitude_prune(model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            weight = param.data
            threshold = np.percentile(weight.cpu().numpy().abs(), pruning_ratio * 100)
            mask = (weight.abs() > threshold).float()
            param.data *= mask
    return model


def prune_and_retrain(model: nn.Module, pruning_ratio: float = 0.3,
                      prune_epochs: int = 5, lr: float = 0.001) -> nn.Module:
    model = magnitude_prune(model, pruning_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(prune_epochs):
        dummy_input = torch.randn(32, model.in_features if hasattr(model, "in_features") else 20)
        dummy_target = torch.randn(32, model.out_features if hasattr(model, "out_features") else 2)
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        optimizer.step()

    return model


def count_zero_weights(model: nn.Module) -> dict:
    stats = {}
    total_zeros = 0
    total_params = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            zeros = (param.data == 0).sum().item()
            total = param.numel()
            total_zeros += zeros
            total_params += total
            stats[name] = {"zeros": zeros, "total": total, "sparsity": zeros / total}
    stats["total"] = {"zeros": total_zeros, "total": total_params,
                       "sparsity": total_zeros / total_params if total_params > 0 else 0}
    return stats


def sparsity_report(model: nn.Module) -> str:
    stats = count_zero_weights(model)
    lines = [f"{'='*50}", f"{'Model Sparsity Report':^50}", f"{'='*50}"]
    lines.append(f"{'Layer':30s} {'Sparsity':>10s}")
    lines.append(f"{'-'*50}")
    for name, s in stats.items():
        if name == "total":
            continue
        lines.append(f"{name:30s} {s['sparsity']:>10.2%}")
    lines.append(f"{'-'*50}")
    t = stats["total"]
    lines.append(f"{'OVERALL':30s} {t['sparsity']:>10.2%}")
    lines.append(f"{'='*50}")
    return "\n".join(lines)


def main():
    import os, yaml, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.neural_network import DriveNetwork

    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("sensor_count", 16) + 4
    model = DriveNetwork(state_dim, 1)
    print(sparsity_report(model))

    for ratio in [0.1, 0.3, 0.5]:
        pruned = DriveNetwork(state_dim, 1)
        pruned.load_state_dict(model.state_dict())
        magnitude_prune(pruned, ratio)
        stats = count_zero_weights(pruned)
        print(f"Pruning ratio {ratio:.0%}: overall sparsity = {stats['total']['sparsity']:.2%}")


if __name__ == "__main__":
    main()
