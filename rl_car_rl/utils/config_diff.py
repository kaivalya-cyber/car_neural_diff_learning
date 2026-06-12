import os
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def diff_configs(config_a: dict, config_b: dict, name_a: str = "A", name_b: str = "B") -> str:
    all_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))
    lines = [f"{'='*70}"]
    lines.append(f"{'Configuration Diff':^70}")
    lines.append(f"{'='*70}")
    lines.append(f"{'Key':30s} {name_a:>18s} {name_b:>18s}")
    lines.append(f"{'-'*70}")

    changes = 0
    for key in all_keys:
        val_a = config_a.get(key, "<MISSING>")
        val_b = config_b.get(key, "<MISSING>")
        if str(val_a) != str(val_b):
            changes += 1
            lines.append(f"{key:30s} {str(val_a):>18s} {str(val_b):>18s}")

    lines.append(f"{'-'*70}")
    lines.append(f"{'Total differences':30s} {changes:>18d}")
    lines.append(f"{'='*70}")
    return "\n".join(lines)


def diff_config_files(path_a: str, path_b: str, name_a: str = "", name_b: str = "") -> str:
    if not name_a:
        name_a = os.path.basename(path_a)
    if not name_b:
        name_b = os.path.basename(path_b)

    try:
        config_a = load_config(path_a)
    except Exception as e:
        return f"Error loading {path_a}: {e}"

    try:
        config_b = load_config(path_b)
    except Exception as e:
        return f"Error loading {path_b}: {e}"

    return diff_configs(config_a, config_b, name_a, name_b)


def main():
    import sys
    if len(sys.argv) < 3:
        paths = ["configs/hyperparameters.yaml"]
        preset_dir = "configs/presets"
        if os.path.isdir(preset_dir):
            for f in sorted(os.listdir(preset_dir)):
                if f.endswith(".yaml"):
                    path_b = os.path.join(preset_dir, f)
                    print(diff_config_files(paths[0], path_b))
                    print()
        return

    path_a = sys.argv[1]
    path_b = sys.argv[2]
    name_a = sys.argv[3] if len(sys.argv) > 3 else ""
    name_b = sys.argv[4] if len(sys.argv) > 4 else ""
    print(diff_config_files(path_a, path_b, name_a, name_b))


if __name__ == "__main__":
    main()
