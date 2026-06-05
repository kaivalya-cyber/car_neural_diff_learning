"""
Self-play competitive training: two policies trained against each other
on the same track, periodically replaced by the stronger opponent.

Usage:
    python main.py --mode self-play
"""

import os
import yaml
import numpy as np
import torch
from env.multi_car_env import MultiCarEnv
from training.trainer import PPOTrainer, Memory


def train_self_play(
    config_overrides=None,
    seed=None,
):
    """Self-play training loop with two competing policies."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, "configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_overrides:
        config.update(config_overrides)

    sensor_count = config.get("sensor_count", 16)
    obstacle_count = config.get("obstacle_count", 0)
    state_dim = sensor_count + 4
    max_episodes = config.get("max_episodes", 2000)
    update_timestep = config.get("update_timestep", 500)

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Self-play training on device: {device}")

    # Two trainers with identical architecture
    trainer1 = PPOTrainer(
        state_dim=state_dim, action_dim=2,
        lr=config["learning_rate"], gamma=config["gamma"],
        lam=config.get("gae_lambda", 0.95), K_epochs=config["k_epochs"],
        eps_clip=config["eps_clip"], max_grad_norm=config.get("max_grad_norm", 0.5),
        lr_decay=config.get("lr_decay", 0.999),
        entropy_coef=config.get("entropy_coef", 0.01),
        entropy_decay=config.get("entropy_decay", 1.0),
        hidden_size=config.get("hidden_size", 256),
        num_blocks=config.get("num_blocks", 2),
        dropout=config.get("dropout", 0.0),
        lr_schedule=config.get("lr_schedule", "exponential"),
        device=str(device),
    )
    trainer2 = PPOTrainer(
        state_dim=state_dim, action_dim=2,
        lr=config["learning_rate"], gamma=config["gamma"],
        lam=config.get("gae_lambda", 0.95), K_epochs=config["k_epochs"],
        eps_clip=config["eps_clip"], max_grad_norm=config.get("max_grad_norm", 0.5),
        lr_decay=config.get("lr_decay", 0.999),
        entropy_coef=config.get("entropy_coef", 0.01),
        entropy_decay=config.get("entropy_decay", 1.0),
        hidden_size=config.get("hidden_size", 256),
        num_blocks=config.get("num_blocks", 2),
        dropout=config.get("dropout", 0.0),
        lr_schedule=config.get("lr_schedule", "exponential"),
        device=str(device),
    )

    env = MultiCarEnv(num_cars=2, sensor_count=sensor_count, obstacle_count=obstacle_count)

    memory1 = Memory()
    memory2 = Memory()
    time_step = 0
    episodes_completed = 0

    # Track performance for replacement decisions
    rewards_1 = []
    rewards_2 = []
    swap_interval = 100  # evaluate and potentially swap every N episodes

    checkpoint_dir = os.path.join(project_dir, "checkpoints", "self_play")
    os.makedirs(checkpoint_dir, exist_ok=True)

    obs = env.reset()
    print(f"Starting self-play training (max {max_episodes} episodes)...")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=max_episodes, desc="Self-Play", unit="ep")
        use_pbar = True
    except ImportError:
        pbar = None
        use_pbar = False

    while episodes_completed < max_episodes:
        time_step += 1

        action1, raw1, logp1, _ = trainer1.policy.get_action(obs[0])
        action2, raw2, logp2, _ = trainer2.policy.get_action(obs[1])
        actions = np.stack([action1, action2])

        next_obs, rewards, dones, infos = env.step(actions)

        # Store in memories (per-car terminals, not combined)
        memory1.states.append(obs[0])
        memory1.actions.append(raw1.detach().cpu().numpy())
        memory1.logprobs.append(logp1.detach().cpu().numpy())
        memory1.rewards.append(np.array([rewards[0] + rewards[1] * 0.1]))
        memory1.is_terminals.append(np.array([dones[0]]))

        memory2.states.append(obs[1])
        memory2.actions.append(raw2.detach().cpu().numpy())
        memory2.logprobs.append(logp2.detach().cpu().numpy())
        memory2.rewards.append(np.array([rewards[1] + rewards[0] * 0.1]))
        memory2.is_terminals.append(np.array([dones[1]]))

        obs = next_obs

        if any(dones):
            episodes_completed += 1
            rewards_1.append(rewards[0])
            rewards_2.append(rewards[1])

            if use_pbar and pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "P1": f"{rewards[0]:.1f}",
                    "P2": f"{rewards[1]:.1f}",
                })
            elif episodes_completed % 10 == 0:
                print(f"Ep {episodes_completed}: P1={rewards[0]:.1f} P2={rewards[1]:.1f}")

            obs = env.reset()

            # Check if weaker player should copy the stronger one
            if episodes_completed % swap_interval == 0 and len(rewards_1) >= swap_interval // 2:
                r1 = np.mean(rewards_1[-50:])
                r2 = np.mean(rewards_2[-50:])
                strong_ckpt = os.path.join(checkpoint_dir, "latest_sp.pth")
                if r1 > r2 * 1.2:
                    trainer1.save(is_best=False, checkpoint_dir=checkpoint_dir)
                    os.replace(os.path.join(checkpoint_dir, "latest.pth"), strong_ckpt)
                    trainer2.load(strong_ckpt)
                    print(f"  Swapped: P1 > P2 ({r1:.1f} vs {r2:.1f}), copied P1->P2")
                elif r2 > r1 * 1.2:
                    trainer2.save(is_best=False, checkpoint_dir=checkpoint_dir)
                    os.replace(os.path.join(checkpoint_dir, "latest.pth"), strong_ckpt)
                    trainer1.load(strong_ckpt)
                    print(f"  Swapped: P2 > P1 ({r2:.1f} vs {r1:.1f}), copied P2->P1")

        # Update both policies
        if time_step > 0 and time_step % update_timestep == 0:
            p_loss1, v_loss1 = trainer1.update(memory1)
            p_loss2, v_loss2 = trainer2.update(memory2)
            memory1.clear_memory()
            memory2.clear_memory()
            time_step = 0

    # Save final policies
    trainer1.save(is_best=False, checkpoint_dir=os.path.join(checkpoint_dir, "player1"))
    trainer2.save(is_best=False, checkpoint_dir=os.path.join(checkpoint_dir, "player2"))
    if use_pbar and pbar is not None:
        pbar.close()

    r1f = np.mean(rewards_1[-50:]) if len(rewards_1) >= 50 else np.mean(rewards_1)
    r2f = np.mean(rewards_2[-50:]) if len(rewards_2) >= 50 else np.mean(rewards_2)
    print(f"Self-play complete. P1 final avg: {r1f:.1f}, P2 final avg: {r2f:.1f}")
    env.close()
