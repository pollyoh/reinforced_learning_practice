"""
Reinforcement Learning Core Components
======================================

This package provides educational implementations of fundamental
reinforcement learning concepts. The implementations are designed
to be simple, well-documented, and beginner-friendly.

Modules:
    - policy: Various policy implementations (epsilon-greedy, softmax, neural)
    - reward: Reward function implementations (simple, shaped, preference-based)
    - learning: Learning algorithms (REINFORCE, PPO, Q-Learning)
    - environment: Simple environments for demonstrations
    - utils: Helper functions for RL workflows

Key Concepts:
    - Policy: A mapping from states to actions (or action probabilities)
    - Reward: Feedback signal that guides learning
    - Value: Expected cumulative future reward from a state
    - Trajectory: A sequence of (state, action, reward) tuples

Example:
    >>> from src.environment import GridWorld
    >>> from src.policy import EpsilonGreedyPolicy
    >>> from src.learning import QLearner
    >>>
    >>> env = GridWorld(size=5)
    >>> policy = EpsilonGreedyPolicy(n_actions=4, epsilon=0.1)
    >>> learner = QLearner(policy, learning_rate=0.1, gamma=0.99)
    >>>
    >>> # Training loop
    >>> for episode in range(100):
    ...     state = env.reset()
    ...     done = False
    ...     while not done:
    ...         action = policy.select_action(state)
    ...         next_state, reward, done = env.step(action)
    ...         learner.update(state, action, reward, next_state, done)
    ...         state = next_state
"""

from src.policy import (
    BasePolicy,
    RandomPolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
    SimpleNeuralPolicy,
)
from src.reward import (
    RewardFunction,
    SimpleReward,
    ShapedReward,
    PreferenceBasedReward,
)
from src.learning import (
    BaseLearner,
    REINFORCELearner,
    SimplePPO,
    QLearner,
)
from src.environment import (
    BaseEnvironment,
    GridWorld,
    SimpleTextEnvironment,
)
from src.utils import (
    collect_trajectory,
    compute_returns,
    compute_advantages,
    plot_learning_curve,
)

__version__ = "0.1.0"
__all__ = [
    # Policies
    "BasePolicy",
    "RandomPolicy",
    "EpsilonGreedyPolicy",
    "SoftmaxPolicy",
    "SimpleNeuralPolicy",
    # Rewards
    "RewardFunction",
    "SimpleReward",
    "ShapedReward",
    "PreferenceBasedReward",
    # Learning algorithms
    "BaseLearner",
    "REINFORCELearner",
    "SimplePPO",
    "QLearner",
    # Environments
    "BaseEnvironment",
    "GridWorld",
    "SimpleTextEnvironment",
    # Utilities
    "collect_trajectory",
    "compute_returns",
    "compute_advantages",
    "plot_learning_curve",
]
