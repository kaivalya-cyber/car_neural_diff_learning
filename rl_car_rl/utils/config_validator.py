import os
import sys
import yaml

EXPECTED_SCHEMA = {
    "learning_rate": {"type": float, "required": True, "min": 1e-6, "max": 1.0},
    "gamma": {"type": float, "required": True, "min": 0.0, "max": 1.0},
    "gae_lambda": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "k_epochs": {"type": int, "required": True, "min": 1, "max": 100},
    "eps_clip": {"type": float, "required": True, "min": 0.0, "max": 1.0},
    "max_grad_norm": {"type": float, "required": False, "min": 0.0, "max": 100.0},
    "lr_decay": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "lr_warmup_epochs": {"type": int, "required": False, "min": 0, "max": 10000},
    "lr_warmup_start_factor": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "entropy_coef": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "entropy_decay": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "normalize_rewards": {"type": bool, "required": False},
    "gradient_accumulation_steps": {"type": int, "required": False, "min": 1, "max": 128},
    "action_repeat": {"type": int, "required": False, "min": 1, "max": 100},
    "ou_sigma": {"type": float, "required": False, "min": 0.0, "max": 10.0},
    "ou_theta": {"type": float, "required": False, "min": 0.0, "max": 10.0},
    "ou_sigma_decay": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "max_episodes": {"type": int, "required": True, "min": 1, "max": 100000},
    "max_steps_per_episode": {"type": int, "required": False, "min": 1, "max": 100000},
    "update_timestep": {"type": int, "required": True, "min": 1, "max": 100000},
    "num_envs": {"type": int, "required": True, "min": 1, "max": 1024},
    "sensor_count": {"type": int, "required": True, "min": 1, "max": 256},
    "sensor_front_density": {"type": float, "required": False, "min": 0.0, "max": 10.0},
    "obstacle_count": {"type": int, "required": False, "min": 0, "max": 100},
    "val_interval": {"type": int, "required": False, "min": 0, "max": 10000},
    "early_stop_patience": {"type": int, "required": False, "min": 0, "max": 100000},
    "early_stop_min_delta": {"type": float, "required": False, "min": 0.0, "max": 100.0},
    "top_k_checkpoints": {"type": int, "required": False, "min": 0, "max": 100},
    "curriculum_start": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "curriculum_min": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "curriculum_max": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "curriculum_up_threshold": {"type": float, "required": False, "min": -1e6, "max": 1e6},
    "curriculum_down_threshold": {"type": float, "required": False, "min": -1e6, "max": 1e6},
    "curriculum_up_rate": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "curriculum_down_rate": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "curriculum_threshold_growth": {"type": float, "required": False, "min": 0.0, "max": 10.0},
    "curriculum_window": {"type": int, "required": False, "min": 1, "max": 10000},
    "curriculum_min_samples": {"type": int, "required": False, "min": 1, "max": 1000},
    "hidden_size": {"type": int, "required": False, "min": 1, "max": 4096},
    "num_blocks": {"type": int, "required": False, "min": 1, "max": 100},
    "dropout": {"type": float, "required": False, "min": 0.0, "max": 1.0},
    "track_type": {"type": str, "required": False, "choices": ["procedural", "oval", "figure_8", "multi_loop"]},
    "lr_schedule": {"type": str, "required": False, "choices": ["exponential", "cosine"]},
    "use_wandb": {"type": bool, "required": False},
    "trace_log_interval": {"type": int, "required": False, "min": 0, "max": 100000},
}


def validate_config(config: dict, schema: dict | None = None) -> list[str]:
    errors = []
    schema = schema or EXPECTED_SCHEMA

    for key, rules in schema.items():
        if rules.get("required") and key not in config:
            errors.append(f"Missing required key: {key}")
            continue
        if key not in config:
            continue

        value = config[key]
        expected_type = rules["type"]

        if rules.get("choices") and value not in rules["choices"]:
            errors.append(
                f"{key}: '{value}' not in allowed choices {rules['choices']}"
            )
        elif expected_type == float and isinstance(value, int):
            config[key] = float(value)
        elif not isinstance(value, expected_type):
            errors.append(
                f"{key}: expected {expected_type.__name__}, got {type(value).__name__} ({value})"
            )

        if rules.get("min") is not None and isinstance(value, (int, float)):
            if value < rules["min"]:
                errors.append(
                    f"{key}: {value} is below minimum {rules['min']}"
                )
        if rules.get("max") is not None and isinstance(value, (int, float)):
            if value > rules["max"]:
                errors.append(
                    f"{key}: {value} is above maximum {rules['max']}"
                )

    return errors


def validate_config_file(path: str) -> tuple[bool, list[str]]:
    if not os.path.exists(path):
        return False, [f"File not found: {path}"]
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML parse error: {e}"]

    errors = validate_config(config)
    return len(errors) == 0, errors


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["configs/hyperparameters.yaml"]
    all_ok = True
    for path in paths:
        ok, errors = validate_config_file(path)
        if ok:
            print(f"[OK] {path}")
        else:
            all_ok = False
            print(f"[FAIL] {path}")
            for e in errors:
                print(f"  - {e}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
