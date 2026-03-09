import argparse
import sys
import yaml
import os
import time

# Ensure packages can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.training_loop import train_agent
from training.tuner import tune_hyperparameters
from training.trainer import PPOTrainer
from env.environment import CarEnv
from visualization.renderer import Renderer

def evaluate():
    config_path = os.path.join("configs", "hyperparameters.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = CarEnv()
    trainer = PPOTrainer(state_dim=9, action_dim=2)
    
    if not trainer.load("checkpoints/latest.pth"):
        print("Running with random weights...")
        
    renderer = Renderer(env, fps=60)
    state = env.reset()
    
    running = True
    print("Starting evaluation display. Close the window to exit.")
    
    while running:
        # Deterministic action
        final_action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
        state, reward, done, info = env.step(final_action)
        running = renderer.render()
        
        if done:
            print(f"Episode Done. Crashed: {info.get('crashed', False)}, Steps: {env.current_step}, Reward: {reward}")
            state = env.reset()
            time.sleep(1)
            
    renderer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Car Agent Execution")
    parser.add_argument('--mode', choices=['train', 'evaluate', 'tune'], default='train', help='Mode to run: train, evaluate, or tune')
    args = parser.parse_args()

    if args.mode == 'train':
        train_agent()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'tune':
        tune_hyperparameters()
