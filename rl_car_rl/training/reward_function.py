def compute_reward(
    velocity: float,
    progress_diff: float,
    crashed: bool,
    done: bool,
    center_distance: float = 0.0,
    steering_change: float = 0.0,
) -> float:
    """
    Compute the reward for the agent at each timestep.

    Rewards progress, speed in the optimal range, centerline position,
    and penalizes crashes and jerky steering.
    """
    reward = 0.0

    # Progress reward (primary signal)
    reward += 50.0 * progress_diff

    # Speed-zone bonus: reward optimal racing speed (30-40), penalize too slow or too fast
    optimal_min, optimal_max = 20.0, 40.0
    if velocity < optimal_min:
        reward += 0.02 * velocity  # encourage going faster
    elif velocity > optimal_max:
        reward -= 0.01 * (velocity - optimal_max)  # penalize excessive speed
    else:
        reward += 0.5  # bonus for being in the sweet spot

    # Centerline bonus: reward staying near track center
    reward += 0.5 * (1.0 - center_distance)

    # Smooth steering penalty
    reward -= 0.05 * steering_change

    # Small time penalty to encourage efficiency
    reward -= 0.01

    # Crash penalty
    if crashed:
        reward -= 10.0

    # Lap completion handled in environment with separate bonus

    return float(reward)
