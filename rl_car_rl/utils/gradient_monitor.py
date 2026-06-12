import torch
import numpy as np


class GradientMonitor:
    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.step_count = 0
        self.history = {"grad_norm": [], "grad_max": [], "grad_mean": [], "param_norm": []}

    def watch(self, model: torch.nn.Module) -> dict:
        self.step_count += 1
        stats = {}
        total_norm = 0.0
        total_max = 0.0
        total_mean = 0.0
        param_norm = 0.0
        count = 0

        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                total_max = max(total_max, p.grad.data.abs().max().item())
                total_mean += p.grad.data.abs().mean().item()
                count += 1
            if p.data is not None:
                param_norm += p.data.norm(2).item() ** 2

        stats["grad_norm"] = np.sqrt(total_norm) if count > 0 else 0.0
        stats["grad_max"] = total_max
        stats["grad_mean"] = total_mean / count if count > 0 else 0.0
        stats["param_norm"] = np.sqrt(param_norm)
        stats["grad_ratio"] = stats["grad_norm"] / (stats["param_norm"] + 1e-8)

        if self.step_count % self.log_freq == 0:
            for k in stats:
                self.history[k].append(stats[k])

        return stats

    def summary(self) -> str:
        lines = [f"{'='*50}", f"{'Gradient Monitor Summary':^50}", f"{'='*50}"]
        for key, values in self.history.items():
            if values:
                arr = np.array(values)
                lines.append(f"{key:20s} | mean={arr.mean():.4e} max={arr.max():.4e}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def check_explosion(self, threshold: float = 10.0) -> bool:
        if self.history["grad_norm"] and self.history["grad_norm"][-1] > threshold:
            return True
        return False

    def check_vanishing(self, threshold: float = 1e-6) -> bool:
        if self.history["grad_norm"] and self.history["grad_norm"][-1] < threshold:
            return True
        return False


def main():
    import os, yaml
    from agent.neural_network import DriveNetwork

    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("sensor_count", 16) + 4
    model = DriveNetwork(state_dim, 1)
    dummy = torch.randn(4, state_dim, requires_grad=True)
    out = model(dummy)
    loss = out.sum()
    loss.backward()

    monitor = GradientMonitor(log_freq=1)
    stats = monitor.watch(model)
    print("Gradient Monitor Test")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:15s}: {v:.6e}")
    print(monitor.summary())


if __name__ == "__main__":
    main()
