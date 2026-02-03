"""
Reward Function Implementations for Reinforcement Learning
==========================================================

The reward function is crucial in RL - it defines what the agent should learn.
    R(s, a, s') = reward for taking action 'a' in state 's' and reaching 's''

Key concepts:
    - Reward hypothesis: All goals can be described as maximizing expected reward
    - Sparse vs Dense rewards:
        - Sparse: Reward only at goal (+1) or failure (-1)
        - Dense: Intermediate rewards to guide learning
    - Reward shaping: Adding intermediate rewards to speed up learning

This module provides:
    1. SimpleReward: Basic +1/-1 rewards
    2. ShapedReward: Rewards with intermediate signals
    3. PreferenceBasedReward: Learning rewards from human preferences (RLHF-style)

Warning - Reward Hacking:
    Agents may find unexpected ways to maximize reward that don't align
    with the designer's intent. This is a key challenge in RL safety.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    A reward function takes a transition (s, a, s') and returns a scalar reward.
    The design of the reward function significantly impacts learning:
    - Too sparse: Agent struggles to learn (no feedback)
    - Too dense: Agent may learn shortcuts (reward hacking)

    Example:
        >>> class MyReward(RewardFunction):
        ...     def compute_reward(self, state, action, next_state, info):
        ...         return 1.0 if next_state == goal else 0.0
    """

    @abstractmethod
    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute the reward for a transition.

        Args:
            state: State before action
            action: Action taken
            next_state: Resulting state
            info: Additional environment information

        Returns:
            Scalar reward value
        """
        pass

    def reset(self) -> None:
        """
        Reset any internal state (called at episode start).

        Override this if your reward function has episode-level state.
        """
        pass


class SimpleReward(RewardFunction):
    """
    Simple reward function with configurable goal and penalty rewards.

    This is the most basic reward structure:
        - Goal reached: +reward_goal
        - Step taken: +reward_step (often negative for time penalty)
        - Invalid action: +reward_invalid

    Example:
        >>> reward_fn = SimpleReward(
        ...     goal_position=(4, 4),
        ...     reward_goal=10.0,
        ...     reward_step=-0.1,
        ...     reward_invalid=-1.0
        ... )
        >>>
        >>> # Agent reaches goal
        >>> r = reward_fn.compute_reward(
        ...     state=(3, 4), action=1, next_state=(4, 4), info={'reached_goal': True}
        ... )
        >>> print(r)  # 10.0
    """

    def __init__(
        self,
        goal_position: Optional[Tuple[int, int]] = None,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_invalid: float = -0.1
    ):
        """
        Initialize simple reward function.

        Args:
            goal_position: Target position (for grid environments)
            reward_goal: Reward for reaching the goal
            reward_step: Reward per step (negative = time penalty)
            reward_invalid: Reward for invalid actions (e.g., hitting wall)
        """
        self.goal_position = goal_position
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_invalid = reward_invalid

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward based on transition.

        Priority:
        1. Check if goal reached
        2. Check if action was invalid
        3. Return step penalty

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            info: Dict with 'reached_goal', 'invalid_action' flags

        Returns:
            Scalar reward
        """
        info = info or {}

        # Check if goal was reached
        if info.get('reached_goal', False):
            return self.reward_goal

        # Check goal position if provided
        if self.goal_position is not None:
            next_pos = tuple(next_state[:2]) if len(next_state) >= 2 else tuple(next_state)
            if next_pos == self.goal_position:
                return self.reward_goal

        # Check if action was invalid
        if info.get('invalid_action', False):
            return self.reward_invalid

        # Default step reward
        return self.reward_step


class ShapedReward(RewardFunction):
    """
    Reward shaping adds intermediate rewards to guide learning.

    The idea: Instead of only rewarding the goal, provide "hints" along the way.

    Potential-based shaping (Ng et al., 1999):
        F(s, s') = gamma * phi(s') - phi(s)

    Where phi(s) is a potential function. This is guaranteed to not change
    the optimal policy while potentially speeding up learning.

    Common shaping approaches:
        - Distance-based: Reward getting closer to goal
        - Progress-based: Reward intermediate milestones
        - Curiosity-based: Reward visiting new states

    Warning:
        Poor reward shaping can lead to suboptimal policies!
        The agent might maximize shaped rewards without achieving the goal.

    Example:
        >>> def distance_potential(state, goal):
        ...     return -np.linalg.norm(state - goal)  # Negative distance
        >>>
        >>> reward_fn = ShapedReward(
        ...     goal_position=(10, 10),
        ...     potential_fn=distance_potential,
        ...     gamma=0.99,
        ...     base_reward=-0.01
        ... )
    """

    def __init__(
        self,
        goal_position: Tuple[int, int],
        potential_fn: Optional[callable] = None,
        gamma: float = 0.99,
        base_reward: float = -0.01,
        goal_reward: float = 10.0,
        shaping_weight: float = 1.0
    ):
        """
        Initialize shaped reward function.

        Args:
            goal_position: Target position
            potential_fn: Custom potential function phi(state, goal) -> float
            gamma: Discount factor for potential-based shaping
            base_reward: Reward per step
            goal_reward: Reward for reaching goal
            shaping_weight: How much to weight the shaping term
        """
        self.goal_position = np.array(goal_position)
        self.gamma = gamma
        self.base_reward = base_reward
        self.goal_reward = goal_reward
        self.shaping_weight = shaping_weight

        # Default potential: negative Manhattan distance
        if potential_fn is None:
            self.potential_fn = self._default_potential
        else:
            self.potential_fn = potential_fn

    def _default_potential(self, state: np.ndarray) -> float:
        """
        Default potential function: negative Manhattan distance to goal.

        Manhattan distance is appropriate for grid worlds where
        movement is restricted to cardinal directions.

        Args:
            state: Current state (position)

        Returns:
            Potential value (higher = closer to goal)
        """
        state_pos = np.array(state[:2]) if len(state) >= 2 else np.array(state)
        manhattan_dist = np.sum(np.abs(state_pos - self.goal_position))
        return -manhattan_dist  # Negative so closer = higher potential

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute shaped reward.

        R_shaped(s, a, s') = R_base(s, a, s') + F(s, s')

        Where F(s, s') = gamma * phi(s') - phi(s)

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            info: Additional information

        Returns:
            Shaped reward value
        """
        info = info or {}

        # Check if goal reached
        next_pos = np.array(next_state[:2]) if len(next_state) >= 2 else np.array(next_state)
        if np.array_equal(next_pos, self.goal_position):
            return self.goal_reward

        if info.get('reached_goal', False):
            return self.goal_reward

        # Compute potential-based shaping
        state_arr = np.array(state)
        next_state_arr = np.array(next_state)

        phi_s = self.potential_fn(state_arr)
        phi_s_next = self.potential_fn(next_state_arr)

        # F(s, s') = gamma * phi(s') - phi(s)
        shaping_reward = self.gamma * phi_s_next - phi_s

        # Total reward = base + shaping
        total_reward = self.base_reward + self.shaping_weight * shaping_reward

        return total_reward


class PreferenceBasedReward(RewardFunction):
    """
    Preference-based reward learning (foundation for RLHF).

    Instead of hand-crafting rewards, learn them from human preferences!

    How it works:
        1. Collect pairs of trajectories (trajectory_A, trajectory_B)
        2. Human indicates which is preferred (or equal)
        3. Train a reward model to predict these preferences
        4. Use the learned reward for RL training

    The Bradley-Terry model for preferences:
        P(A > B) = exp(R(A)) / (exp(R(A)) + exp(R(B)))
                 = sigmoid(R(A) - R(B))

    Where R(trajectory) = sum of rewards for each transition.

    This is the foundation of RLHF (Reinforcement Learning from Human Feedback)
    used to train language models like ChatGPT.

    Example:
        >>> reward_fn = PreferenceBasedReward(state_dim=4, hidden_size=32)
        >>>
        >>> # Collect preferences
        >>> traj_a = [(s1, a1, s2), (s2, a2, s3)]  # Trajectory A
        >>> traj_b = [(s1, b1, s4), (s4, b2, s5)]  # Trajectory B
        >>> preference = 1  # Human prefers A
        >>>
        >>> # Update reward model
        >>> reward_fn.update_from_preference(traj_a, traj_b, preference)
        >>>
        >>> # Now use learned reward
        >>> r = reward_fn.compute_reward(state, action, next_state)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 32,
        learning_rate: float = 0.001,
        seed: Optional[int] = None
    ):
        """
        Initialize preference-based reward model.

        Args:
            state_dim: Dimension of state space
            hidden_size: Hidden layer size for reward network
            learning_rate: Learning rate for updates
            seed: Random seed
        """
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Simple neural network for reward prediction
        # Input: state (or state-action pair)
        # Output: scalar reward
        self._init_network()

        # Store preferences for batch training
        self.preferences: List[Tuple] = []

    def _init_network(self) -> None:
        """
        Initialize reward network weights.

        Architecture: state -> hidden -> reward
        """
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (self.state_dim + self.hidden_size))
        self.W1 = self.rng.normal(0, scale1, (self.state_dim, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        scale2 = np.sqrt(2.0 / (self.hidden_size + 1))
        self.W2 = self.rng.normal(0, scale2, (self.hidden_size, 1))
        self.b2 = np.zeros(1)

    def _forward(self, state: np.ndarray) -> float:
        """
        Forward pass through reward network.

        Args:
            state: Input state

        Returns:
            Predicted reward
        """
        state = np.asarray(state).flatten()

        # Hidden layer with tanh activation
        h = np.tanh(state @ self.W1 + self.b1)

        # Output layer (no activation - reward can be any value)
        reward = float((h @ self.W2 + self.b2)[0])

        return reward

    def compute_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward using learned reward model.

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            info: Additional information

        Returns:
            Learned reward value
        """
        # Use next_state as input (reward for reaching this state)
        return self._forward(next_state)

    def compute_trajectory_reward(
        self,
        trajectory: List[Tuple[np.ndarray, int, np.ndarray]]
    ) -> float:
        """
        Compute total reward for a trajectory.

        Args:
            trajectory: List of (state, action, next_state) tuples

        Returns:
            Sum of rewards along trajectory
        """
        total = 0.0
        for state, action, next_state in trajectory:
            total += self.compute_reward(state, action, next_state)
        return total

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with numerical stability."""
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1 + exp_x)

    def update_from_preference(
        self,
        trajectory_a: List[Tuple[np.ndarray, int, np.ndarray]],
        trajectory_b: List[Tuple[np.ndarray, int, np.ndarray]],
        preference: int
    ) -> float:
        """
        Update reward model from a preference.

        Uses Bradley-Terry model:
            P(A > B) = sigmoid(R(A) - R(B))

        Loss = -log(P(preferred > other))

        Args:
            trajectory_a: First trajectory
            trajectory_b: Second trajectory
            preference: 1 if A preferred, -1 if B preferred, 0 if equal

        Returns:
            Loss value
        """
        # Compute trajectory rewards
        reward_a = self.compute_trajectory_reward(trajectory_a)
        reward_b = self.compute_trajectory_reward(trajectory_b)

        # Compute probability under Bradley-Terry model
        if preference == 1:  # A preferred
            prob = self._sigmoid(reward_a - reward_b)
            target = 1.0
        elif preference == -1:  # B preferred
            prob = self._sigmoid(reward_b - reward_a)
            target = 1.0
        else:  # Equal preference
            prob = 0.5  # For equal, we don't update much
            target = 0.5

        # Compute loss (negative log likelihood)
        loss = -np.log(prob + 1e-10)

        # Gradient update (simplified - full version would use backprop)
        # For educational purposes, we do a simple update
        if preference != 0:
            gradient = target - prob

            # Update for trajectory A states
            for state, action, next_state in trajectory_a:
                self._update_weights(next_state, gradient * preference)

            # Update for trajectory B states (opposite direction)
            for state, action, next_state in trajectory_b:
                self._update_weights(next_state, -gradient * preference)

        return loss

    def _update_weights(self, state: np.ndarray, gradient: float) -> None:
        """
        Update network weights for a single state.

        Args:
            state: Input state
            gradient: Gradient to apply
        """
        state = np.asarray(state).flatten()

        # Forward pass to get intermediate values
        z1 = state @ self.W1 + self.b1
        h = np.tanh(z1)

        # Backward pass
        # Gradient through output layer
        grad_W2 = h.reshape(-1, 1) * gradient * self.learning_rate
        grad_b2 = gradient * self.learning_rate

        # Gradient through tanh
        grad_h = self.W2.flatten() * gradient
        grad_z1 = grad_h * (1 - h**2)  # tanh derivative

        # Gradient through input layer
        grad_W1 = np.outer(state, grad_z1) * self.learning_rate
        grad_b1 = grad_z1 * self.learning_rate

        # Apply updates
        self.W2 += grad_W2
        self.b2 += grad_b2
        self.W1 += grad_W1
        self.b1 += grad_b1

    def add_preference(
        self,
        trajectory_a: List[Tuple[np.ndarray, int, np.ndarray]],
        trajectory_b: List[Tuple[np.ndarray, int, np.ndarray]],
        preference: int
    ) -> None:
        """
        Store a preference for batch training.

        Args:
            trajectory_a: First trajectory
            trajectory_b: Second trajectory
            preference: 1 (A better), -1 (B better), 0 (equal)
        """
        self.preferences.append((trajectory_a, trajectory_b, preference))

    def train_on_preferences(self, n_epochs: int = 10) -> List[float]:
        """
        Train on all stored preferences.

        Args:
            n_epochs: Number of training epochs

        Returns:
            List of average losses per epoch
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            # Shuffle preferences
            indices = self.rng.permutation(len(self.preferences))

            for idx in indices:
                traj_a, traj_b, pref = self.preferences[idx]
                loss = self.update_from_preference(traj_a, traj_b, pref)
                epoch_loss += loss

            avg_loss = epoch_loss / len(self.preferences) if self.preferences else 0
            losses.append(avg_loss)

        return losses

