import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module, name: str = "Model", input_size: tuple | None = None) -> str:
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"{name:^80}")
    lines.append(f"{'='*80}")
    lines.append(f"{'Layer':40s} {'Output Shape':30s} {'Params':10s}")
    lines.append(f"{'-'*80}")

    total_params = 0
    for mod_name, module in model.named_modules():
        if not list(module.children()):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                total_params += params
                out_shape = _infer_output_shape(module, input_size) if input_size else "?"
                lines.append(f"{mod_name:40s} {str(out_shape):30s} {params:<10,}")

    total, trainable = count_parameters(model)
    lines.append(f"{'-'*80}")
    lines.append(f"{'Total params':40s} {total:<10,}")
    lines.append(f"{'Trainable params':40s} {trainable:<10,}")
    lines.append(f"{'Non-trainable params':40s} {total - trainable:<10,}")
    lines.append(f"{'='*80}")
    return "\n".join(lines)


def _infer_output_shape(module, input_size):
    try:
        dummy = torch.randn(1, *input_size)
        with torch.no_grad():
            out = module(dummy)
        return list(out.shape)
    except Exception:
        return "?"


def print_model_comparison(models: dict[str, nn.Module], input_size: tuple | None = None):
    lines = []
    lines.append(f"{'='*100}")
    lines.append(f"{'Model Comparison':^100}")
    lines.append(f"{'='*100}")
    lines.append(f"{'Name':25s} {'Total Params':15s} {'Trainable':15s} {'Size (MB)':15s}")
    lines.append(f"{'-'*100}")
    for name, model in models.items():
        total, trainable = count_parameters(model)
        size_mb = total * 4 / (1024 * 1024)
        lines.append(f"{name:25s} {total:<15,} {trainable:<15,} {size_mb:<15.2f}")
    lines.append(f"{'='*100}")
    return "\n".join(lines)


def main():
    from agent.neural_network import DriveNetwork
    from agent.policy import RacingPolicy
    import yaml
    import os

    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hidden = config.get("hidden_size", 256)
    blocks = config.get("num_blocks", 2)
    dropout = config.get("dropout", 0.0)
    state_dim = config.get("sensor_count", 16) + 4

    device = "cpu"
    policy = RacingPolicy(state_dim, 2, device, hidden_size=hidden, num_blocks=blocks, dropout=dropout)
    value_net = DriveNetwork(state_dim, 1, hidden_size=hidden, num_blocks=blocks, dropout=dropout)

    print(model_summary(policy.net, "Policy Network (Actor)", (state_dim,)))
    print()
    print(model_summary(value_net, "Value Network (Critic)", (state_dim,)))
    print()
    print(print_model_comparison({
        "Policy (Actor)": policy.net,
        "Value (Critic)": value_net,
    }, (state_dim,)))


if __name__ == "__main__":
    main()
