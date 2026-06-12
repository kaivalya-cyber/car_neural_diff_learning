"""
Model profiling: parameter count, inference speed benchmark.

Usage:
    python profile_model.py --checkpoint checkpoints/best.pth
    python profile_model.py --checkpoint checkpoints/best.pth --batch-sizes 1,32,128
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def benchmark_inference(
    model: torch.nn.Module,
    state_dim: int,
    batch_sizes: list[int],
    num_warmup: int = 20,
    num_runs: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark inference latency across batch sizes."""
    model.eval()
    model.to(device)
    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, state_dim, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_runs) * 1000
        fps = num_runs / elapsed
        results[f"batch_{bs}_ms"] = avg_ms
        results[f"batch_{bs}_fps"] = fps

    return results


def profile_model(
    checkpoint_path: str,
    state_dim: int = 20,
    batch_sizes: list[int] = None,
    device: str = None,
):
    """Run full model profiling and print report."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling on device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    policy_weights = checkpoint.get("policy_net", {})

    hidden_size = 256
    for key in policy_weights:
        if "weight" in key:
            hidden_size = policy_weights[key].shape[0]
            break

    log_std = checkpoint.get("policy_log_std")
    action_dim = 2 if log_std is None else log_std.shape[-1]

    from agent.neural_network import DriveNetwork
    policy_net = DriveNetwork(state_dim, action_dim, hidden_size=hidden_size)
    policy_net.load_state_dict(policy_weights)

    value_net = DriveNetwork(state_dim, 1, hidden_size=hidden_size)
    if "value_net" in checkpoint:
        value_net.load_state_dict(checkpoint["value_net"])

    # Parameter counts
    policy_params = count_parameters(policy_net)
    value_params = count_parameters(value_net)

    print(f"\n{'='*50}")
    print(f"Model Profile: {os.path.basename(checkpoint_path)}")
    print(f"{'='*50}")
    print(f"  Architecture: hidden={hidden_size}, state_dim={state_dim}")
    print(f"\n  Parameter Counts:")
    print(f"    Policy network: {policy_params['total']:,} ({policy_params['trainable']:,} trainable)")
    print(f"    Value network:  {value_params['trainable']:,} ({value_params['trainable']:,} trainable)")
    print(f"    Total:          {policy_params['total'] + value_params['total']:,}")

    # Inference benchmarks
    print(f"\n  Inference Benchmarks ({device}):")
    print(f"    {'Batch':<10} {'Latency (ms)':>14} {'Throughput (fps)':>18}")
    print(f"    {'-'*10} {'-'*14} {'-'*18}")

    bench = benchmark_inference(policy_net, state_dim, batch_sizes, device=device)
    for bs in batch_sizes:
        ms = bench[f"batch_{bs}_ms"]
        fps = bench[f"batch_{bs}_fps"]
        print(f"    {bs:<10} {ms:>14.3f} {fps:>18.1f}")

    # Model size
    tmp_path = "/tmp/profile_temp.pth"
    torch.save({"policy_net": policy_weights, "policy_log_std": log_std}, tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    print(f"\n  Checkpoint size: {model_size_mb:.1f} MB")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model profiling and benchmarking")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Checkpoint path")
    parser.add_argument("--state-dim", type=int, default=20, help="Observation dimension")
    parser.add_argument("--batch-sizes", default="1,8,32,128", help="Comma-separated batch sizes")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    profile_model(args.checkpoint, args.state_dim, batch_sizes, args.device)
