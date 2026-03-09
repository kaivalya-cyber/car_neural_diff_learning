import time
import sys
import os

# Ensure the parent directory is in sys.path so env/training packages can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import CarEnv
from training.trainer import PPOTrainer
from visualization.renderer import Renderer

def evaluate():
    env = CarEnv()
    trainer = PPOTrainer(state_dim=7, action_dim=2)
    
    try:
        trainer.load("checkpoints/latest.pth")
        print("Loaded trained model.")
    except Exception as e:
        print(f"Could not load model: {e}. Running with random weights.")
        
    renderer = Renderer(env, fps=60)
    state = env.reset()
    
    running = True
    print("Starting evaluation display. Close the window to exit.")
    
    while running:
        # Evaluate deterministically
        final_action, _, _, _ = trainer.policy.get_action(state, deterministic=True)
        
        state, reward, done, info = env.step(final_action)
        running = renderer.render()
        
        if done:
            print(f"Episode Done. Crashed: {info.get('crashed', False)}, Steps: {env.current_step}")
            state = env.reset()
            time.sleep(1)
            
    renderer.close()

if __name__ == "__main__":
    evaluate()
