def compute_reward(velocity: float, progress_diff: float, crashed: bool, done: bool) -> float:
    # Phase 5: Implement Reward System
    reward = 0.0
    
    # Forward motion
    reward += 1.0 * velocity * 0.01  # scale down slightly
    
    # Track progress reward
    # Depending on how progress diff is measured, usually something like +500 * progress_diff
    reward += 50.0 * progress_diff
    
    # Time penalty
    reward -= 0.01
    
    # Terminal rewards
    if crashed:
        reward -= 10.0
    
    # To handle leaving the track, standard checking usually treats it as a crash
    
    return float(reward)
