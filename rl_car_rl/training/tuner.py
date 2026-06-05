import yaml
import os
import numpy as np
import torch
import pandas as pd
from datetime import datetime

from env.vector_env import VectorEnv
from training.trainer import PPOTrainer, Memory
from training.curriculum import CurriculumManager


def sample_config(search_space: dict) -> dict:
    """Sample one configuration from the search space."""
    config = {}
    for key, value in search_space.items():
        if isinstance(value, list):
            choice = np.random.choice(value)
            # Preserve original type: int stays int, float stays float
            if isinstance(value[0], int):
                config[key] = int(choice)
            else:
                config[key] = float(choice)
        elif isinstance(value, dict) and "min" in value and "max" in value:
            if isinstance(value["min"], int) and isinstance(value["max"], int):
                config[key] = int(np.random.randint(value["min"], value["max"] + 1))
            else:
                config[key] = float(np.random.uniform(value["min"], value["max"]))
        else:
            config[key] = value
    return config


def run_tuning_session(config: dict, run_id: int) -> dict:
    num_envs = 16
    sensor_count = 16
    state_dim = sensor_count + 4
    max_episodes = 250
    update_timestep = 2000

    print(f"\n[{run_id}] Testing Config: {config}")
    env = VectorEnv(num_envs=num_envs, sensor_count=sensor_count)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    lr = config.get("learning_rate", 3e-4)
    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=2,
        lr=lr,
        gamma=config.get("gamma", 0.99),
        lam=config.get("gae_lambda", 0.95),
        K_epochs=config.get("k_epochs", 4),
        eps_clip=config.get("eps_clip", 0.2),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        lr_decay=config.get("lr_decay", 0.999),
        lr_warmup_epochs=config.get("lr_warmup_epochs", 0),
        entropy_coef=config.get("entropy_coef", 0.01),
        entropy_decay=config.get("entropy_decay", 1.0),
        normalize_rewards=config.get("normalize_rewards", True),
        hidden_size=config.get("hidden_size", 256),
        num_blocks=config.get("num_blocks", 2),
        dropout=config.get("dropout", 0.0),
        device=str(device),
    )

    memory = Memory()
    curriculum = CurriculumManager()
    env.set_difficulty(curriculum.get_generation_params())

    time_step = 0
    episodes_completed = 0
    current_ep_rewards = np.zeros(num_envs)
    recent_rewards = []

    state = env.reset()

    while episodes_completed < max_episodes:
        time_step += 1
        final_action, raw_action, log_prob, _ = trainer.policy.get_action(
            state, deterministic=False
        )
        next_state, reward, done, _ = env.step(final_action)

        memory.states.append(state)
        memory.actions.append(raw_action.detach().cpu().numpy())
        memory.logprobs.append(log_prob.detach().cpu().numpy())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        state = next_state
        current_ep_rewards += reward

        for i in range(num_envs):
            if done[i]:
                episodes_completed += 1
                recent_rewards.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0

                if episodes_completed >= max_episodes:
                    break

        if time_step > 0 and time_step % update_timestep == 0:
            if len(recent_rewards) > 0:
                for r in recent_rewards:
                    curriculum.add_reward(r)
                new_level = curriculum.update()
                if new_level != curriculum.level:
                    env.set_difficulty(curriculum.get_generation_params())
                recent_rewards = []

        if time_step > 0 and time_step % (update_timestep // 2) == 0:
            if len(memory.states) > 0:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=trainer.device
                )
                with torch.no_grad():
                    final_value = trainer.value_net(state_tensor).squeeze(-1)
            else:
                final_value = None
            trainer.update(memory, final_value=final_value)
            memory.clear_memory()
            time_step = 0

    env.close()

    final_reward = (
        np.mean(recent_rewards[-50:])
        if len(recent_rewards) >= 50
        else np.mean(recent_rewards) if recent_rewards else 0.0
    )
    print(
        f"[{run_id}] Finished! Reward: {final_reward:.2f} | "
        f"Level: {curriculum.level:.2f}"
    )

    return {
        **config,
        "final_reward": float(final_reward),
        "curriculum_level": float(curriculum.level),
    }


def tune_hyperparameters(budget: int = 16, config_overrides: dict | None = None):
    """
    Run random search hyperparameter tuning.

    Args:
        budget: Number of configurations to try.
        config_overrides: Overrides for the search space.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load search space from tune.yaml
    tune_path = os.path.join(project_dir, "configs", "tune.yaml")
    with open(tune_path, "r") as f:
        search_space = yaml.safe_load(f)

    # Load base config for defaults
    base_path = os.path.join(project_dir, "configs", "hyperparameters.yaml")
    with open(base_path, "r") as f:
        base_config = yaml.safe_load(f)

    if config_overrides:
        search_space.update(config_overrides)

    results = []
    print(f"Starting random search tuning (budget={budget})...")

    for i in range(budget):
        config = sample_config(search_space)
        # Fill in missing values from base config
        for key, value in base_config.items():
            if key not in config:
                config[key] = value
        try:
            res = run_tuning_session(config, i)
            results.append(res)
        except Exception as e:
            print(f"[{i}] Failed: {e}")

    if not results:
        print("No successful runs!")
        return

    # Sort and save
    df = pd.DataFrame(results)
    df = df.sort_values(by=["curriculum_level", "final_reward"], ascending=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(project_dir, "configs", "tuning_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{timestamp}.yaml")

    with open(out_path, "w") as f:
        f.write(f"# Random Search Tuning Results ({timestamp})\n")
        f.write(f"# Budget: {budget}\n\n")

        best = df.iloc[0].to_dict()
        f.write("best_config:\n")
        for k, v in best.items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\ntop_5:\n")
        for idx, (_, row) in enumerate(df.head(5).iterrows()):
            f.write(f"  - rank_{idx + 1}:\n")
            for k, v in row.to_dict().items():
                f.write(f"      {k}: {v}\n")

        f.write(f"\nall_runs:\n")
        for idx, (_, row) in enumerate(df.iterrows()):
            f.write(f"  - run_{idx}:\n")
            for k, v in row.to_dict().items():
                f.write(f"      {k}: {v}\n")

    print(f"Tuning complete! {len(df)} runs saved to {out_path}.")
    print(f"Best config: reward={best['final_reward']:.2f}, level={best['curriculum_level']:.2f}")
