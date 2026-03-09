import yaml
import os
import numpy as np
import time
from env.vector_env import VectorEnv
from training.trainer import PPOTrainer, Memory
from training.curriculum import CurriculumManager
from torch.utils.tensorboard import SummaryWriter
import torch

def train_agent():
    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    num_envs = config.get('num_envs', 64)
    env = VectorEnv(num_envs=num_envs)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    trainer = PPOTrainer(
        state_dim=9, 
        action_dim=2, 
        lr=config['learning_rate'], 
        gamma=config['gamma'], 
        K_epochs=config['k_epochs'],
        eps_clip=config['eps_clip'],
        device=str(device)
    )
    
    memory = Memory()
    
    # Setup TensorBoard and CSV logging
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir="logs/train")
    csv_path = os.path.join("logs", "metrics.csv")
    
    # Init CSV header
    with open(csv_path, "w") as f:
        f.write("episode,reward,length,crash_rate,difficulty\n")
        
    curriculum = CurriculumManager()
    
    # Initialize env with start difficulty
    env.set_difficulty(curriculum.get_generation_params())
    
    max_episodes = config['max_episodes']
    update_timestep = config['update_timestep']
    time_step = 0
    best_reward = -np.inf
    
    # We track episodes by env
    current_ep_rewards = np.zeros(num_envs)
    current_ep_lengths = np.zeros(num_envs)
    episodes_completed = 0
    recent_rewards = []
    
    print(f"Starting Training Loop with {num_envs} vectorized environments...")
    
    state = env.reset()
    start_time = time.time()
    
    while episodes_completed < max_episodes:
        time_step += 1
        
        final_action, raw_action, log_prob, mean = trainer.policy.get_action(state, deterministic=False)
        next_state, reward, done, info = env.step(final_action)
        
        # Store transition (save batched numpy arrays)
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
                # Log metrics for this completed episode
                episodes_completed += 1
                recent_rewards.append(current_ep_rewards[i])
                
                writer.add_scalar('Reward/Episode', current_ep_rewards[i], episodes_completed)
                writer.add_scalar('Length/Episode', current_ep_lengths[i], episodes_completed)
                crash_rate = 1.0 if info[i].get('crashed', False) else 0.0
                writer.add_scalar('Metrics/CrashRate', crash_rate, episodes_completed)
                
                # Append to CSV
                with open(csv_path, "a") as f:
                    f.write(f"{episodes_completed},{current_ep_rewards[i]:.4f},{current_ep_lengths[i]},{crash_rate},{curriculum.level:.4f}\n")
                
                # Terminal print
                if episodes_completed % 10 == 0:
                    fps = int((time_step * num_envs) / (time.time() - start_time))
                    print(f"Ep {episodes_completed} | Reward: {current_ep_rewards[i]:.2f} | FPS: {fps}")
                
                # Checkpoints
                if current_ep_rewards[i] > best_reward:
                    best_reward = current_ep_rewards[i]
                    trainer.save(is_best=True)
                    print(f"Saved new best model @ Reward {best_reward:.2f}")
                    
                if episodes_completed % 50 == 0:
                    trainer.save(is_best=False)
                
                # Reset tracking for this env
                current_ep_rewards[i] = 0
                current_ep_lengths[i] = 0
                
                if episodes_completed >= max_episodes:
                    break
        
        # Periodic Curriculum Update
        if time_step > 0 and time_step % (update_timestep * 2) == 0:
            if len(recent_rewards) > 0:
                mean_reward = np.mean(recent_rewards)
                old_level = curriculum.level
                new_level = curriculum.update(mean_reward)
                
                writer.add_scalar('Metrics/DifficultyLevel', new_level, episodes_completed)
                
                if new_level > old_level:
                    print(f"Curriculum Level Upgraded to {new_level:.2f}! (Mean Reward: {mean_reward:.2f})")
                    env.set_difficulty(curriculum.get_generation_params())
                    
                # Clear recent memory for next curriculum observation period
                recent_rewards = []
        
        # Update policy
        if time_step > 0 and time_step % update_timestep == 0:
            print("Optimizing Policy...")
            p_loss, v_loss = trainer.update(memory)
            writer.add_scalar('Loss/Policy', p_loss, episodes_completed)
            writer.add_scalar('Loss/Value', v_loss, episodes_completed)
            
            memory.clear_memory()
            time_step = 0
            
    print("Training Complete.")
    env.close()
