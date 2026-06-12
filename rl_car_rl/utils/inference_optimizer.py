import torch
import torch.nn as nn
import time
import os


def optimize_for_inference(model: nn.Module, input_size: int, device: str = "cpu") -> nn.Module:
    model.eval()
    example = torch.randn(1, input_size).to(device)

    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    traced = torch.jit.freeze(traced)

    return traced


def benchmark(model: nn.Module, input_size: int, device: str = "cpu",
              num_warmup: int = 100, num_runs: int = 1000) -> dict:
    model.eval()
    example = torch.randn(1, input_size).to(device)

    for _ in range(num_warmup):
        _ = model(example)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model(example)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "total_time": elapsed,
        "mean_ms": (elapsed / num_runs) * 1000,
        "fps": num_runs / elapsed,
        "num_runs": num_runs,
    }


def optimize_checkpoint(checkpoint_path: str, state_dim: int, output_path: str = "exported/model_optimized.pt") -> str:
    from agent.neural_network import DriveNetwork

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = DriveNetwork(state_dim, 1)
    if "value_net" in ckpt:
        model.load_state_dict(ckpt["value_net"])
    elif "policy_net" in ckpt:
        policy_state = {}
        for k, v in ckpt["policy_net"].items():
            if k.startswith("fc"):
                new_k = k.replace("fc", "layers.")
            else:
                new_k = k
            policy_state[new_k] = v
        try:
            model.load_state_dict(policy_state, strict=False)
        except Exception:
            model.load_state_dict(ckpt["policy_net"], strict=False)

    traced = optimize_for_inference(model, state_dim)
    torch.jit.save(traced, output_path)
    return output_path


def main():
    import argparse, sys, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/latest.pth")
    parser.add_argument("--output", default="exported/model_optimized.pt")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_dim = config.get("sensor_count", 16) + 4

    from agent.neural_network import DriveNetwork
    model = DriveNetwork(state_dim, 1)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "value_net" in ckpt:
        model.load_state_dict(ckpt["value_net"])

    print("Benchmarking original model...")
    orig = benchmark(model, state_dim)
    print(f"  Mean: {orig['mean_ms']:.4f}ms | FPS: {orig['fps']:.1f}")

    optimized = optimize_for_inference(model, state_dim)
    print("Benchmarking optimized (TorchScript) model...")
    opt = benchmark(optimized, state_dim)
    print(f"  Mean: {opt['mean_ms']:.4f}ms | FPS: {opt['fps']:.1f}")

    speedup = orig["mean_ms"] / opt["mean_ms"]
    print(f"\nSpeedup: {speedup:.2f}x")

    output_path = optimize_checkpoint(args.checkpoint, state_dim, args.output)
    print(f"Saved optimized model to {output_path}")


if __name__ == "__main__":
    main()
