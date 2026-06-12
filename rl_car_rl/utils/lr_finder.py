import torch
import torch.nn as nn
import numpy as np
import os
import yaml


class LRRangeTest:
    def __init__(self, model: nn.Module, input_size: int, batch_size: int = 64,
                 start_lr: float = 1e-7, end_lr: float = 10.0, num_iter: int = 100):
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
        self.criterion = nn.MSELoss()
        self.results = {"lrs": [], "losses": []}

    def run(self) -> dict:
        lr_mult = (self.end_lr / self.start_lr) ** (1.0 / (self.num_iter - 1))

        for i in range(self.num_iter):
            lr = self.start_lr * (lr_mult ** i)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            dummy_input = torch.randn(self.batch_size, self.input_size)
            dummy_target = torch.randn(self.batch_size, 1)

            self.optimizer.zero_grad()
            output = self.model(dummy_input)
            loss = self.criterion(output, dummy_target)
            loss.backward()
            self.optimizer.step()

            self.results["lrs"].append(lr)
            self.results["losses"].append(loss.item())

        return self._analyze()

    def _analyze(self) -> dict:
        losses = np.array(self.results["losses"])
        lrs = np.array(self.results["lrs"])

        smooth_loss = np.convolve(losses, np.ones(5) / 5, mode="valid")
        if len(smooth_loss) < 2:
            return {"recommended_lr": self.start_lr, "min_lr": self.start_lr, "max_lr": self.end_lr}

        grad = np.gradient(smooth_loss)
        min_grad_idx = np.argmin(grad)
        recommended_lr = lrs[min_grad_idx + 2]

        return {
            "recommended_lr": float(recommended_lr),
            "min_lr": float(lrs[0]),
            "max_lr": float(lrs[-1]),
            "loss_min": float(losses.min()),
            "loss_at_min_grad": float(smooth_loss[min_grad_idx]),
        }

    def plot(self, save_path: str | None = "lr_finder_plot.png"):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.semilogx(self.results["lrs"], self.results["losses"])
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("LR Range Test")
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Saved plot to {save_path}")
            plt.close()
        except ImportError:
            pass


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.neural_network import DriveNetwork

    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("sensor_count", 16) + 4
    model = DriveNetwork(state_dim, 1)
    model.train()

    finder = LRRangeTest(model, state_dim)
    results = finder.run()
    finder.plot()

    print("=" * 50)
    print("LR Range Test Results")
    print("=" * 50)
    print(f"  Recommended LR: {results['recommended_lr']:.2e}")
    print(f"  Search Range:   {results['min_lr']:.2e} - {results['max_lr']:.2e}")
    print(f"  Min Loss:       {results['loss_min']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
