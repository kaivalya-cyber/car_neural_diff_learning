import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "metrics.csv",
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Live Agent Training Dashboard", fontsize=16)
(ax1, ax2), (ax3, ax4) = axes


def animate(i):
    if not os.path.exists(CSV_PATH):
        return

    try:
        data = pd.read_csv(CSV_PATH)
        if len(data) == 0:
            return

        episodes = data["episode"]
        rewards = data["reward"]
        difficulty = data["difficulty"]
        crash_rate = data["crash_rate"]
        center_distance = data.get("center_distance", pd.Series([0.0] * len(data)))
        laps = data.get("laps", pd.Series([0] * len(data)))

        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        # Panel 1: Rewards
        rolling_rewards = rewards.rolling(window=100, min_periods=1).mean()
        ax1.plot(episodes, rewards, label="Reward", alpha=0.25, color="blue")
        ax1.plot(
            episodes, rolling_rewards, label="100-Ep MA", color="red", linewidth=2
        )
        ax1.set_title("Agent Return per Episode")
        ax1.set_ylabel("Reward")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Curriculum Progression
        ax2.plot(
            episodes, difficulty, label="Difficulty Level", color="green", linewidth=2
        )
        ax2.set_title("Curriculum Difficulty")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Level")
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(episodes, 0, difficulty, alpha=0.1, color="green")

        # Panel 3: Crash Rate
        rolling_crash = crash_rate.rolling(window=100, min_periods=1).mean()
        ax3.plot(
            episodes,
            crash_rate,
            label="Crash Rate",
            alpha=0.2,
            color="red",
            linewidth=0.5,
        )
        ax3.plot(
            episodes,
            rolling_crash,
            label="100-Ep MA",
            color="darkred",
            linewidth=2,
        )
        ax3.set_title("Crash Rate")
        ax3.set_ylabel("Rate")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Center Distance & Laps
        rolling_center = center_distance.rolling(window=100, min_periods=1).mean()
        ax4.plot(
            episodes,
            center_distance,
            label="Center Dist",
            alpha=0.25,
            color="purple",
            linewidth=0.5,
        )
        ax4.plot(
            episodes,
            rolling_center,
            label="100-Ep MA",
            color="purple",
            linewidth=2,
        )
        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            episodes, laps, label="Laps", color="orange", linewidth=2, alpha=0.8
        )
        ax4.set_title("Center Distance & Laps")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Center Distance", color="purple")
        ax4_twin.set_ylabel("Laps", color="orange")
        ax4.grid(True, alpha=0.3)
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

        plt.tight_layout()

    except Exception as e:
        print(f"Warning: Dashboard error: {e}")


ani = FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)

if __name__ == "__main__":
    print("Launching Live Training Dashboard...")
    print("Keep this window open and start the training loop in another terminal.")
    plt.tight_layout()
    plt.show()
