from pathlib import Path
import sys

sys.path.insert(0, 'rl_car_rl')
from training.training_loop import train_agent


def main():
    out_dir = Path('experiments/20260312_smoke_test')
    train_agent(
        config_overrides={'max_episodes': 5, 'num_envs': 2},
        output_dir=str(out_dir),
        use_curriculum=True,
        seed=123,
    )
    print(f"smoke test complete: {out_dir}")


if __name__ == '__main__':
    main()
