"""
Utility Functions for Reinforcement Learning
=============================================

This module provides helper functions commonly used in RL:
    - collect_trajectory(): Gather experience from environment
    - compute_returns(): Calculate discounted returns
    - compute_advantages(): Estimate advantages for policy gradient
    - plot_learning_curve(): Visualize training progress

These functions are the "glue" between environments, policies, and learning algorithms.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


def collect_trajectory(
    env: Any,
    policy: Any,
    max_steps: Optional[int] = None,
    render: bool = False
) -> Dict[str, List]:
    """
    Collect a single trajectory (episode) from the environment.

    A trajectory is a sequence of experiences:
        (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)

    This is the fundamental data collection step in RL.

    Args:
        env: Environment with reset() and step() methods
        policy: Policy with select_action() method
        max_steps: Maximum steps (uses env.max_steps if None)
        render: Whether to print environment state

    Returns:
        Dictionary with:
            - 'states': List of states
            - 'actions': List of actions
            - 'rewards': List of rewards
            - 'next_states': List of next states
            - 'dones': List of done flags
            - 'infos': List of info dicts

    Example:
        >>> from src.environment import GridWorld
        >>> from src.policy import RandomPolicy
        >>>
        >>> env = GridWorld(size=5)
        >>> policy = RandomPolicy(n_actions=4)
        >>> trajectory = collect_trajectory(env, policy)
        >>>
        >>> print(f"Episode length: {len(trajectory['rewards'])}")
        >>> print(f"Total reward: {sum(trajectory['rewards']):.2f}")
    """
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': [],
        'infos': []
    }

    # Reset environment
    state = env.reset()
    done = False
    steps = 0

    # Determine max steps
    if max_steps is None:
        max_steps = getattr(env, 'max_steps', 1000)

    while not done and steps < max_steps:
        # Render if requested
        if render:
            print(env.render())
            print()

        # Select and execute action
        action = policy.select_action(state)
        next_state, reward, done, info = env.step(action)

        # Store transition
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['next_states'].append(next_state)
        trajectory['dones'].append(done)
        trajectory['infos'].append(info)

        # Move to next state
        state = next_state
        steps += 1

    return trajectory


def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
    normalize: bool = False
) -> List[float]:
    """
    Compute discounted returns for each timestep.

    The return G_t is the discounted sum of future rewards:
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
            = r_t + gamma * G_{t+1}

    Why discount?
        - Immediate rewards are more certain than future rewards
        - Prevents infinite returns in continuing tasks
        - gamma close to 1: far-sighted agent
        - gamma close to 0: myopic agent

    Efficient computation (backwards):
        G_T = r_T
        G_{T-1} = r_{T-1} + gamma * G_T
        G_{T-2} = r_{T-2} + gamma * G_{T-1}
        ...

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (0 to 1)
        normalize: Whether to normalize returns (zero mean, unit variance)

    Returns:
        List of returns [G_0, G_1, ..., G_T]

    Example:
        >>> rewards = [1, 0, 0, 10]  # Sparse reward at end
        >>> returns = compute_returns(rewards, gamma=0.9)
        >>> # G_3 = 10
        >>> # G_2 = 0 + 0.9 * 10 = 9
        >>> # G_1 = 0 + 0.9 * 9 = 8.1
        >>> # G_0 = 1 + 0.9 * 8.1 = 8.29
        >>> print(returns)  # [8.29, 8.1, 9.0, 10.0]
    """
    if not rewards:
        return []

    returns = [0.0] * len(rewards)

    # Compute returns backwards (more efficient)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    # Optionally normalize
    if normalize and len(returns) > 1:
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.tolist()

    return returns


def compute_advantages(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    normalize: bool = True
) -> List[float]:
    """
    Compute Generalized Advantage Estimation (GAE).

    The advantage A(s,a) tells us how much better action 'a' is
    compared to the average action in state 's'.

    GAE balances bias and variance in advantage estimation:
        - lambda=0: One-step TD (low variance, high bias)
        - lambda=1: Monte Carlo (high variance, low bias)
        - lambda=0.95: Good balance (commonly used)

    Formula:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  # TD error
        A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...

    Why GAE?
        - Simple advantages have high variance (noisy gradients)
        - GAE smooths out the variance while keeping useful signal
        - Critical for stable policy gradient training

    Args:
        rewards: List of rewards
        values: List of value estimates V(s_t)
        gamma: Discount factor
        lambda_gae: GAE lambda parameter
        normalize: Whether to normalize advantages

    Returns:
        List of advantage estimates

    Example:
        >>> rewards = [1, 0, 0, 10]
        >>> values = [5, 4, 3, 8]  # Value estimates from critic
        >>> advantages = compute_advantages(rewards, values, gamma=0.99, lambda_gae=0.95)
    """
    if not rewards or not values:
        return []

    n = len(rewards)
    advantages = [0.0] * n

    # Need V(s_{T+1}) = 0 for terminal state
    values = list(values) + [0.0]

    # Compute GAE backwards
    running_advantage = 0.0
    for t in reversed(range(n)):
        # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        running_advantage = delta + gamma * lambda_gae * running_advantage
        advantages[t] = running_advantage

    # Normalize advantages (helps with training stability)
    if normalize and len(advantages) > 1:
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.tolist()

    return advantages


def plot_learning_curve(
    rewards_history: List[float],
    window_size: int = 10,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> str:
    """
    Create ASCII plot of learning progress (no matplotlib dependency).

    For visualization with matplotlib, see plot_learning_curve_matplotlib().

    Args:
        rewards_history: List of episode rewards
        window_size: Window for moving average
        title: Plot title
        save_path: Path to save plot (as text file)

    Returns:
        ASCII art representation of learning curve

    Example:
        >>> rewards = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        >>> print(plot_learning_curve(rewards))
        # Learning Curve
        # Episode 10 | #################### | 45.0
        # Moving Avg | #################### | 22.5
    """
    if not rewards_history:
        return "No data to plot"

    n_episodes = len(rewards_history)
    min_reward = min(rewards_history)
    max_reward = max(rewards_history)

    # Compute moving average
    moving_avg = []
    for i in range(len(rewards_history)):
        start = max(0, i - window_size + 1)
        window = rewards_history[start:i + 1]
        moving_avg.append(sum(window) / len(window))

    # ASCII plot parameters
    width = 50  # Chart width

    def reward_to_bar(reward: float) -> str:
        """Convert reward to ASCII bar."""
        if max_reward == min_reward:
            normalized = 0.5
        else:
            normalized = (reward - min_reward) / (max_reward - min_reward)
        bar_length = int(normalized * width)
        return '#' * bar_length + ' ' * (width - bar_length)

    # Build output
    lines = [
        title,
        "=" * (width + 30),
        f"Episodes: {n_episodes}",
        f"Min reward: {min_reward:.2f}",
        f"Max reward: {max_reward:.2f}",
        f"Final avg ({window_size} ep): {moving_avg[-1]:.2f}",
        "-" * (width + 30),
    ]

    # Sample episodes to display (at most 20)
    if n_episodes <= 20:
        sample_indices = list(range(n_episodes))
    else:
        step = n_episodes // 20
        sample_indices = list(range(0, n_episodes, step))
        if sample_indices[-1] != n_episodes - 1:
            sample_indices.append(n_episodes - 1)

    for i in sample_indices:
        bar = reward_to_bar(rewards_history[i])
        avg = moving_avg[i]
        lines.append(f"Ep {i+1:4d} | {bar} | {rewards_history[i]:7.2f} (avg: {avg:.2f})")

    lines.append("-" * (width + 30))

    result = '\n'.join(lines)

    # Save if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(result)

    return result


def epsilon_schedule(
    episode: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    decay_episodes: int = 500
) -> float:
    """
    Compute epsilon for epsilon-greedy with linear decay.

    Epsilon controls exploration in epsilon-greedy:
        - High epsilon: More exploration (random actions)
        - Low epsilon: More exploitation (greedy actions)

    Linear decay schedule:
        - Start with high epsilon (explore a lot)
        - Gradually decrease (exploit more as we learn)
        - Floor at epsilon_end (always some exploration)

    Args:
        episode: Current episode number
        epsilon_start: Initial epsilon
        epsilon_end: Final epsilon (floor)
        decay_episodes: Episodes over which to decay

    Returns:
        Epsilon value for this episode

    Example:
        >>> for ep in range(1000):
        ...     epsilon = epsilon_schedule(ep, epsilon_start=1.0, epsilon_end=0.01)
        ...     # Use epsilon in policy
    """
    if episode >= decay_episodes:
        return epsilon_end

    # Linear interpolation
    progress = episode / decay_episodes
    epsilon = epsilon_start - progress * (epsilon_start - epsilon_end)
    return epsilon


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Reproducibility is important for:
        - Debugging (reproduce exact same run)
        - Fair comparison of algorithms
        - Scientific validity of results

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    np.random.seed(seed)


def running_mean(values: List[float], window: int = 100) -> List[float]:
    """
    Compute running mean with given window size.

    Useful for smoothing noisy reward curves.

    Args:
        values: List of values
        window: Window size for averaging

    Returns:
        List of running mean values

    Example:
        >>> noisy_rewards = [10, 5, 8, 12, 3, 15, 7, 9]
        >>> smooth_rewards = running_mean(noisy_rewards, window=3)
    """
    if not values:
        return []

    means = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        means.append(sum(window_values) / len(window_values))

    return means


def soft_update(target_params: np.ndarray, source_params: np.ndarray, tau: float = 0.01) -> np.ndarray:
    """
    Soft update for target networks (used in DQN, DDPG, etc.).

    Instead of directly copying parameters, we slowly blend:
        target = tau * source + (1 - tau) * target

    This provides more stable learning in algorithms with target networks.

    Args:
        target_params: Current target network parameters
        source_params: Source network parameters
        tau: Blending factor (0 to 1, typically small like 0.01)

    Returns:
        Updated target parameters

    Example:
        >>> target_weights = np.array([1.0, 2.0, 3.0])
        >>> source_weights = np.array([2.0, 3.0, 4.0])
        >>> updated = soft_update(target_weights, source_weights, tau=0.1)
        >>> # updated = 0.1 * [2,3,4] + 0.9 * [1,2,3] = [1.1, 2.1, 3.1]
    """
    return tau * source_params + (1 - tau) * target_params


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance score.

    This metric is useful for evaluating value function quality:
        EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Interpretation:
        - EV = 1: Perfect predictions
        - EV = 0: Predictions no better than mean
        - EV < 0: Predictions worse than mean

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance score

    Example:
        >>> returns = np.array([10, 20, 30, 40])  # True returns
        >>> value_estimates = np.array([12, 18, 32, 38])  # Value function predictions
        >>> ev = explained_variance(value_estimates, returns)
        >>> print(f"Explained variance: {ev:.2f}")
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0

    return 1 - np.var(y_true - y_pred) / var_true

