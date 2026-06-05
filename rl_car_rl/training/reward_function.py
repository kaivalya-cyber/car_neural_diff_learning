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

    Args:
        velocity: Current car velocity (m/s).
        progress_diff: Change in normalized track progress since last step.
        crashed: Whether the car has collided with a track boundary.
        done: Whether the episode has terminated.
        center_distance: Normalized distance from track centerline [0, 1].
        steering_change: Magnitude of steering change from previous action.

    Returns:
        Scalar reward value.
    """
    reward = 0.0

    # Forward motion bonus (scaled)
    reward += 1.0 * velocity * 0.01

    # Track progress reward
    reward += 50.0 * progress_diff

    # Centerline bonus: reward staying near the center of the track
    # center_distance is [0, 1] where 0 = center, 1 = edge
    reward += 0.5 * (1.0 - center_distance)

    # Smooth steering penalty to discourage jerky driving
    reward -= 0.05 * steering_change

    # Small time penalty to encourage efficiency
    reward -= 0.01

    # Terminal rewards
    if crashed:
        reward -= 10.0

    return float(reward)
