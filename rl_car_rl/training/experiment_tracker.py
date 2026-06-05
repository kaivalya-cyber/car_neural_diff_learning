"""
Experiment tracking: records config, git hash, and final metrics as JSON.
"""

import json
import os
import subprocess
from datetime import datetime


def get_git_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


class ExperimentTracker:
    """Records experiment metadata for reproducibility and comparison."""

    def __init__(self, output_dir: str, config: dict | None = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.start_time = datetime.now()
        self.metadata = {
            "experiment_id": self.start_time.strftime("%Y%m%d_%H%M%S"),
            "start_time": self.start_time.isoformat(),
            "git_hash": get_git_hash(),
            "git_branch": get_git_branch(),
            "config": config or {},
            "metrics": {},
            "checkpoints": [],
            "end_time": None,
            "duration_seconds": None,
            "status": "running",
        }

    def record_checkpoint(self, reward: float, episode: int, path: str) -> None:
        """Record a saved checkpoint."""
        self.metadata["checkpoints"].append(
            {"reward": float(reward), "episode": episode, "path": path}
        )

    def record_metrics(self, metrics: dict) -> None:
        """Update final metrics."""
        self.metadata["metrics"].update(metrics)

    def finalize(self, status: str = "completed") -> str:
        """Mark experiment as complete and save metadata JSON."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()
        self.metadata["status"] = status

        path = os.path.join(self.output_dir, "experiment.json")
        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return path


def load_experiment(path: str) -> dict:
    """Load experiment metadata from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def list_experiments(experiments_dir: str = "experiments") -> list[dict]:
    """List all recorded experiments with key metrics."""
    if not os.path.exists(experiments_dir):
        return []
    results = []
    for root, _, files in os.walk(experiments_dir):
        if "experiment.json" in files:
            try:
                exp = load_experiment(os.path.join(root, "experiment.json"))
                results.append(
                    {
                        "id": exp.get("experiment_id", "unknown"),
                        "status": exp.get("status", "unknown"),
                        "best_reward": exp.get("metrics", {}).get("best_reward"),
                        "episodes": exp.get("metrics", {}).get("episodes_completed"),
                        "duration": exp.get("duration_seconds"),
                        "path": root,
                    }
                )
            except Exception:
                pass
    return sorted(results, key=lambda x: x.get("best_reward") or -float("inf"), reverse=True)
