import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# Define path to metrics.csv
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "metrics.csv")

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Live Agent Training Dashboard', fontsize=16)

def animate(i):
    if not os.path.exists(CSV_PATH):
        return
        
    try:
        data = pd.read_csv(CSV_PATH)
        
        if len(data) == 0:
            return
            
        # Clear current axes
        ax1.clear()
        ax2.clear()
        
        episodes = data['episode']
        rewards = data['reward']
        difficulty = data['difficulty']
        
        # Setup rolling mean (100 episodes) for rewards
        rolling_rewards = rewards.rolling(window=100, min_periods=1).mean()
        
        # Plot 1: Rewards
        ax1.plot(episodes, rewards, label='Reward', alpha=0.3, color='blue')
        ax1.plot(episodes, rolling_rewards, label='100-Ep Moving Avg', color='red', linewidth=2)
        ax1.set_title('Agent Return (Reward) per Episode')
        ax1.set_ylabel('Reward')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Curriculum Progression
        ax2.plot(episodes, difficulty, label='Difficulty Level', color='green', linewidth=2)
        ax2.set_title('Curriculum Difficulty Progression (0=Easy, 1=Hard)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Difficulty Level')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(episodes, 0, difficulty, alpha=0.1, color='green')
        
    except Exception as e:
        print(f"Warning: Failed to read dashboard CSV: {e}")

ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False) # update every 1s

if __name__ == "__main__":
    print("Launching Live Training Dashboard...")
    print("Keep this window open and start the training loop in another terminal.")
    plt.tight_layout()
    plt.show()
