import yaml
import os
import numpy as np
import time
from env.vector_env import VectorEnv
from training.trainer import PPOTrainer, Memory
from training.curriculum import CurriculumManager
from training.experiment_tracker import ExperimentTracker
from torch.utils.tensorboard import SummaryWriter
import torch


def run_validation_episode(trainer, device, sensor_count, obstacle_count=0):
    """Run one deterministic evaluation episode on a single environment."""
    from env.environment import CarEnv

    eval_env = CarEnv(sensor_count=sensor_count, obstacle_count=obstacle_count)
    state = eval_env.reset()
    total_reward = 0.0
    steps = 0
    crashed = False
    laps = 0

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        final_action, _, _, _ = trainer.policy.get_action(
            state_tensor.cpu().numpy().squeeze(0), deterministic=True
        )
        state, reward, done, info = eval_env.step(final_action)
        total_reward += reward
        steps += 1
        laps = info.get("lap_count", 0)
        if done:
            crashed = info.get("crashed", False)
            break

    return total_reward, steps, crashed, laps


def train_agent(
    config_overrides=None,
    output_dir=None,
    use_curriculum=True,
    seed=None,
    resume=False,
):
    """
    Main training loop for the RL car agent.

    Args:
        config_overrides: Dict of hyperparameter overrides.
        output_dir: Directory for logs and checkpoints.
        use_curriculum: Whether to use curriculum learning.
        seed: Random seed for reproducibility.
        resume: Whether to resume from the latest checkpoint.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, "configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config_overrides:
        config.update(config_overrides)

    num_envs = config.get("num_envs", 64)
    sensor_count = config.get("sensor_count", 16)
    obstacle_count = config.get("obstacle_count", 0)
    state_dim = sensor_count + 4  # sensors + velocity, heading, angular_vel, center_dist

    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    env = VectorEnv(num_envs=num_envs, sensor_count=sensor_count, obstacle_count=obstacle_count)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")
    print(f"State dim: {state_dim} ({sensor_count} sensors + 4 car features)")

    trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=2,
        lr=config["learning_rate"],
        gamma=config["gamma"],
        lam=config.get("gae_lambda", 0.95),
        K_epochs=config["k_epochs"],
        eps_clip=config["eps_clip"],
        max_grad_norm=config.get("max_grad_norm", 0.5),
        lr_decay=config.get("lr_decay", 0.999),
        entropy_coef=config.get("entropy_coef", 0.01),
        entropy_decay=config.get("entropy_decay", 1.0),
        lr_warmup_epochs=config.get("lr_warmup_epochs", 0),
        lr_warmup_start_factor=config.get("lr_warmup_start_factor", 0.1),
        normalize_rewards=config.get("normalize_rewards", True),
        ou_sigma=config.get("ou_sigma", 0.0),
        ou_theta=config.get("ou_theta", 0.15),
        ou_sigma_decay=config.get("ou_sigma_decay", 1.0),
        hidden_size=config.get("hidden_size", 256),
        num_blocks=config.get("num_blocks", 2),
        dropout=config.get("dropout", 0.0),
        device=str(device),
    )

    # --- Resume from checkpoint ---
    start_episode = 0
    best_reward = -np.inf
    base_dir = output_dir or project_dir
    checkpoint_dir = os.path.join(base_dir, "checkpoints")

    if resume:
        latest_path = os.path.join(checkpoint_dir, "latest.pth")
        if os.path.exists(latest_path):
            trainer.load(latest_path)
            print("Resumed from latest checkpoint.")
        else:
            print("No checkpoint found; starting fresh.")

    memory = Memory()

    # Setup experiment tracking
    tracker = ExperimentTracker(output_dir=logs_dir, config=config)

    # Setup TensorBoard and CSV logging
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logs_dir, "train"))

    # Determine CSV write mode
    csv_path = os.path.join(logs_dir, "metrics.csv")
    csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    if resume and csv_exists:
        try:
            with open(csv_path, "r") as f:
                lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line:
                    start_episode = int(last_line.split(",")[0])
            import pandas as pd

            df = pd.read_csv(csv_path)
            if not df.empty:
                best_reward = float(df["reward"].max())
        except Exception:
            start_episode = 0
    else:
        with open(csv_path, "w") as f:
            f.write(
                "episode,reward,length,crash_rate,difficulty,center_distance,laps\n"
            )

    curriculum = CurriculumManager(
        start_level=config.get("curriculum_start", 0.0),
        min_level=config.get("curriculum_min", 0.0),
        max_level=config.get("curriculum_max", 1.0),
        increase_threshold=config.get("curriculum_up_threshold", 40.0),
        decrease_threshold=config.get("curriculum_down_threshold", 10.0),
        increase_rate=config.get("curriculum_up_rate", 0.05),
        decrease_rate=config.get("curriculum_down_rate", 0.03),
        threshold_growth=config.get("curriculum_threshold_growth", 1.01),
        window_size=config.get("curriculum_window", 10),
        min_samples=config.get("curriculum_min_samples", 5),
    )

    if use_curriculum:
        env.set_difficulty(curriculum.get_generation_params())

    max_episodes = config["max_episodes"]
    update_timestep = config["update_timestep"]

    # --- Early stopping ---
    early_stop_patience = config.get("early_stop_patience", 200)
    early_stop_min_delta = config.get("early_stop_min_delta", 0.5)
    last_improvement_episode = start_episode

    # --- Top-K checkpoint management ---
    top_k = config.get("top_k_checkpoints", 1)
    top_k_rewards = []  # list of (reward, episode) tuples

    # --- Periodic validation ---
    val_interval = config.get("val_interval", 50)
    eval_sensor_count = sensor_count  # must match for checkpoint compatibility

    time_step = 0
    global_step = 0

    current_ep_rewards = np.zeros(num_envs)
    current_ep_lengths = np.zeros(num_envs)
    episodes_completed = start_episode
    recent_rewards = []

    print(
        f"Starting Training Loop with {num_envs} vectorized environments "
        f"(resuming from episode {episodes_completed})..."
    )

    # Initialize progress bar if tqdm is available
    try:
        from tqdm import tqdm
        pbar = tqdm(total=max_episodes, initial=episodes_completed, desc="Training", unit="ep")
        use_pbar = True
    except ImportError:
        pbar = None
        use_pbar = False

    state = env.reset()
    start_time = time.time()

    while episodes_completed < max_episodes:
        time_step += 1
        global_step += 1

        final_action, raw_action, log_prob, mean = trainer.policy.get_action(
            state, deterministic=False
        )
        next_state, reward, done, info = env.step(final_action)

        memory.states.append(state)
        memory.actions.append(raw_action.detach().cpu().numpy())
        memory.logprobs.append(log_prob.detach().cpu().numpy())
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        state = next_state
        current_ep_rewards += reward
        current_ep_lengths += 1

        for i in range(num_envs):
            if done[i]:
                episodes_completed += 1
                recent_rewards.append(current_ep_rewards[i])

                writer.add_scalar(
                    "Reward/Episode", current_ep_rewards[i], episodes_completed
                )
                writer.add_scalar(
                    "Length/Episode", current_ep_lengths[i], episodes_completed
                )
                crash_rate = 1.0 if info[i].get("crashed", False) else 0.0
                writer.add_scalar(
                    "Metrics/CrashRate", crash_rate, episodes_completed
                )
                center_dist = info[i].get("center_distance", 0.0)
                writer.add_scalar(
                    "Metrics/CenterDistance", center_dist, episodes_completed
                )
                laps_done = info[i].get("lap_count", 0)
                writer.add_scalar(
                    "Metrics/Laps", laps_done, episodes_completed
                )

                with open(csv_path, "a") as f:
                    f.write(
                        f"{episodes_completed},{current_ep_rewards[i]:.4f},"
                        f"{current_ep_lengths[i]},{crash_rate},"
                        f"{curriculum.level:.4f},{center_dist:.4f},{laps_done}\n"
                    )

                if episodes_completed % 10 == 0:
                    fps = int(
                        (time_step * num_envs)
                        / max(time.time() - start_time, 0.001)
                    )
                    current_lr = trainer.policy_scheduler.get_last_lr()[0]
                    
                    if use_pbar and pbar is not None:
                        pbar.update(10)
                        pbar.set_postfix({
                            "R": f"{current_ep_rewards[i]:.1f}",
                            "FPS": fps,
                            "LR": f"{current_lr:.2e}",
                        })
                    else:
                        print(
                            f"Ep {episodes_completed} | "
                            f"Reward: {current_ep_rewards[i]:.2f} | "
                            f"FPS: {fps} | "
                            f"LR: {current_lr:.2e} | "
                            f"Ent: {trainer.entropy_coef:.4f}"
                        )

                # Checkpoint saving
                if current_ep_rewards[i] > best_reward + early_stop_min_delta:
                    best_reward = current_ep_rewards[i]
                    last_improvement_episode = episodes_completed
                    
                    # Top-K management: save with episode prefix
                    if top_k > 0:
                        path = trainer.save(
                            is_best=True, checkpoint_dir=checkpoint_dir,
                            episode=episodes_completed
                        )
                        tracker.record_checkpoint(
                            current_ep_rewards[i], episodes_completed, path
                        )
                        top_k_rewards.append((current_ep_rewards[i], episodes_completed))
                        top_k_rewards.sort(key=lambda x: x[0], reverse=True)
                        top_k_rewards = top_k_rewards[:top_k]
                        # Clean up old checkpoints
                        existing = [
                            f for f in os.listdir(checkpoint_dir)
                            if f.startswith("best_ep") and f.endswith(".pth")
                        ]
                        keep_eps = {ep for _, ep in top_k_rewards}
                        for fname in existing:
                            try:
                                ep = int(fname.replace("best_ep", "").replace(".pth", ""))
                                if ep not in keep_eps:
                                    os.remove(os.path.join(checkpoint_dir, fname))
                            except ValueError:
                                pass
                    else:
                        trainer.save(is_best=True, checkpoint_dir=checkpoint_dir)
                    
                    print(
                        f"Saved new best model @ Reward {best_reward:.2f} "
                        f"(ep {episodes_completed})"
                    )

                if episodes_completed % 50 == 0:
                    trainer.save(is_best=False, checkpoint_dir=checkpoint_dir)

                # --- Periodic validation ---
                if (
                    val_interval > 0
                    and episodes_completed % val_interval == 0
                ):
                    val_reward, val_steps, val_crashed, val_laps = (
                        run_validation_episode(
                            trainer, device, eval_sensor_count, obstacle_count
                        )
                    )
                    writer.add_scalar(
                        "Validation/Reward", val_reward, episodes_completed
                    )
                    writer.add_scalar(
                        "Validation/Steps", val_steps, episodes_completed
                    )
                    writer.add_scalar(
                        "Validation/Crashed", float(val_crashed), episodes_completed
                    )
                    writer.add_scalar(
                        "Validation/Laps", val_laps, episodes_completed
                    )
                    print(
                        f"  Validation Ep {episodes_completed} | "
                        f"Reward: {val_reward:.2f} | "
                        f"Steps: {val_steps} | "
                        f"Crashed: {val_crashed} | "
                        f"Laps: {val_laps}"
                    )

                # Reset tracking for this env
                current_ep_rewards[i] = 0
                current_ep_lengths[i] = 0
                # Reset OU noise at episode boundary for cleaner exploration
                trainer.policy.reset_noise()

                if episodes_completed >= max_episodes:
                    break

        # Early stopping check
        if episodes_completed - last_improvement_episode >= early_stop_patience:
            print(
                f"Early stopping triggered after {early_stop_patience} "
                f"episodes without improvement (ep {episodes_completed}). "
                f"Best reward: {best_reward:.2f}"
            )
            break

        # Periodic Curriculum Update
        if (
            use_curriculum
            and global_step > 0
            and global_step % (update_timestep * 2) == 0
        ):
            if len(recent_rewards) > 0:
                mean_reward = np.mean(recent_rewards)
                old_level = curriculum.level
                new_level = curriculum.update()

                writer.add_scalar(
                    "Metrics/DifficultyLevel", new_level, episodes_completed
                )
                stats = curriculum.get_stats()
                writer.add_scalar(
                    "Curriculum/RollingMean", stats["rolling_mean"], episodes_completed
                )
                writer.add_scalar(
                    "Curriculum/Threshold", stats["increase_threshold"], episodes_completed
                )

                if new_level > old_level:
                    print(
                        f"Curriculum Level ↑ to {new_level:.2f}! "
                        f"(Mean: {mean_reward:.2f})"
                    )
                    env.set_difficulty(curriculum.get_generation_params())
                elif new_level < old_level:
                    print(
                        f"Curriculum Level ↓ to {new_level:.2f}! "
                        f"(Mean: {mean_reward:.2f})"
                    )
                    env.set_difficulty(curriculum.get_generation_params())

                # Add all recent rewards to curriculum window
                for r in recent_rewards:
                    curriculum.add_reward(r)
                recent_rewards = []

        # Update policy
        if time_step > 0 and time_step % update_timestep == 0:
            if not use_pbar:
                print("Optimizing Policy...")
            if len(memory.states) > 0 and state is not None:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=trainer.device
                )
                with torch.no_grad():
                    final_value = trainer.value_net(state_tensor).squeeze(-1)
            else:
                final_value = None
            p_loss, v_loss = trainer.update(memory, final_value=final_value)
            writer.add_scalar("Loss/Policy", p_loss, episodes_completed)
            writer.add_scalar("Loss/Value", v_loss, episodes_completed)

            memory.clear_memory()
            time_step = 0

    trainer.save(is_best=False, checkpoint_dir=checkpoint_dir)
    if use_pbar and pbar is not None:
        pbar.close()
    
    # Finalize experiment tracking
    tracker.record_metrics({
        "best_reward": float(best_reward),
        "episodes_completed": episodes_completed,
        "final_curriculum_level": float(curriculum.level),
        "early_stopped": episodes_completed - last_improvement_episode >= early_stop_patience,
    })
    tracker.finalize()
    
    print(
        f"Training Complete. Best reward: {best_reward:.2f}, "
        f"Episodes: {episodes_completed}"
    )
    env.close()
