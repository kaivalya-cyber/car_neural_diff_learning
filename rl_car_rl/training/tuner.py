import yaml
import os
import itertools
import numpy as np
import time
import torch
import pandas as pd

from env.vector_env import VectorEnv
from training.trainer import PPOTrainer, Memory
from training.curriculum import CurriculumManager


def get_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))


def run_tuning_session(config, run_id):
    num_envs = 32
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

    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=2,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        K_epochs=config["k_epochs"],
        eps_clip=config["eps_clip"],
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

        # Curriculum Update
        if time_step > 0 and time_step % update_timestep == 0:
            if len(recent_rewards) > 0:
                mean_reward = np.mean(recent_rewards)
                new_level = curriculum.update(mean_reward)
                if new_level > curriculum.level:
                    env.set_difficulty(curriculum.get_generation_params())
                recent_rewards = []

        # Policy Update with GAE bootstrapping
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
        else np.mean(recent_rewards)
    )
    print(
        f"[{run_id}] Finished! Reward: {final_reward:.2f} | "
        f"Curriculum Level: {curriculum.level:.2f}"
    )

    return {
        **config,
        "final_reward": final_reward,
        "curriculum_level": curriculum.level,
    }


def tune_hyperparameters():
    tune_path = os.path.join("configs", "tune.yaml")
    with open(tune_path, "r") as f:
        grid = yaml.safe_load(f)

    results = []

    for i, config in enumerate(get_combinations(grid)):
        res = run_tuning_session(config, i)
        results.append(res)

    # Sort by metric
    df = pd.DataFrame(results)
    df = df.sort_values(by=["curriculum_level", "final_reward"], ascending=False)

    out_path = os.path.join("configs", "tuning_results.yaml")

    # Save as YAML
    with open(out_path, "w") as f:
        f.write("# Automated Hyperparameter Tuning Results\n")
        f.write("# Sorted best to worst\n\n")

        best_config = df.iloc[0].to_dict()
        f.write("best_config:\n")
        for k, v in best_config.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nall_runs:\n")
        for i, row in df.iterrows():
            f.write(f"  - run_{i}:\n")
            for k, v in row.to_dict().items():
                f.write(f"      {k}: {v}\n")

    print(
        f"Tuning complete! Saved {len(df)} configurations to {out_path}."
    )
