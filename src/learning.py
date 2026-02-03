"""
Learning Algorithm Implementations for Reinforcement Learning
=============================================================

Learning algorithms define HOW the agent improves its policy.

Two main approaches:
    1. Value-based: Learn value function, derive policy (Q-Learning)
    2. Policy-based: Directly optimize the policy (REINFORCE, PPO)

Key concepts:
    - Value function V(s): Expected return starting from state s
    - Action-value Q(s,a): Expected return taking action a in state s
    - Advantage A(s,a) = Q(s,a) - V(s): How much better is action a than average
    - Policy gradient: Gradient of expected return w.r.t. policy parameters

This module provides:
    1. QLearner: Classic value-based method (tabular)
    2. REINFORCELearner: Basic policy gradient method
    3. SimplePPO: Proximal Policy Optimization (simplified)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.policy import BasePolicy, EpsilonGreedyPolicy, SimpleNeuralPolicy
from src.utils import compute_returns, compute_advantages


class BaseLearner(ABC):
    """
    Abstract base class for learning algorithms.

    A learner updates a policy based on experience (transitions).

    Types of learning:
        - Online: Update after each transition
        - Batch: Collect data, then update
        - Episode: Update at end of each episode

    Example:
        >>> class MyLearner(BaseLearner):
        ...     def update(self, transition):
        ...         # Update policy based on transition
        ...         pass
        ...     def train_episode(self, episode_data):
        ...         # Train on complete episode
        ...         pass
    """

    def __init__(self, policy: BasePolicy):
        """
        Initialize learner with a policy.

        Args:
            policy: The policy to optimize
        """
        self.policy = policy

    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Update policy from a single transition.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            Dictionary of metrics (e.g., loss, td_error)
        """
        pass

    def train_episode(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Train on a complete episode (for episodic algorithms).

        Override this for algorithms that need complete episodes
        (e.g., REINFORCE, PPO).

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards

        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Override for episodic algorithms")


class QLearner(BaseLearner):
    """
    Q-Learning: Classic value-based reinforcement learning.

    Q-Learning learns the optimal action-value function Q*(s,a) using
    temporal difference (TD) learning.

    The update rule:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
                                    \___________ TD target __________/
                                                    \__________ TD error _____/

    Key properties:
        - Off-policy: Can learn from any data (exploration policy != learned policy)
        - Tabular: Stores Q-value for each (state, action) pair
        - Guaranteed convergence (under certain conditions)

    Components:
        - Q-table: Matrix of Q(s,a) values
        - Learning rate (alpha): How fast to update (0.1 typical)
        - Discount factor (gamma): How much to value future rewards (0.99 typical)

    Example:
        >>> policy = EpsilonGreedyPolicy(n_actions=4, epsilon=0.1)
        >>> learner = QLearner(
        ...     policy=policy,
        ...     n_states=100,
        ...     n_actions=4,
        ...     learning_rate=0.1,
        ...     gamma=0.99
        ... )
        >>>
        >>> # Training loop
        >>> state = env.reset()
        >>> action = policy.select_action(state)
        >>> next_state, reward, done = env.step(action)
        >>> learner.update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        policy: EpsilonGreedyPolicy,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        initial_q: float = 0.0
    ):
        """
        Initialize Q-Learning.

        Args:
            policy: Epsilon-greedy policy for action selection
            n_states: Number of states (for tabular Q-function)
            n_actions: Number of actions
            learning_rate: Learning rate alpha (how fast to update)
            gamma: Discount factor (how much to value future rewards)
            initial_q: Initial Q-values (optimistic init can help exploration)
        """
        super().__init__(policy)
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Q-table
        # Shape: (n_states, n_actions)
        self.q_table = np.full((n_states, n_actions), initial_q)

        # Share Q-table with policy
        if hasattr(policy, 'q_values'):
            policy.q_values = self.q_table

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> Dict[str, float]:
        """
        Perform one Q-learning update.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether episode ended

        Returns:
            Dictionary with 'td_error' metric
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Compute TD target
        if done:
            # No future rewards if episode ended
            td_target = reward
        else:
            # r + gamma * max_a' Q(s', a')
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q

        # TD error (how wrong our estimate was)
        td_error = td_target - current_q

        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error

        return {'td_error': abs(td_error)}

    def get_value(self, state: int) -> float:
        """
        Get the value of a state: V(s) = max_a Q(s,a)

        Args:
            state: State index

        Returns:
            State value
        """
        return float(np.max(self.q_table[state]))

    def get_best_action(self, state: int) -> int:
        """
        Get the greedy action for a state.

        Args:
            state: State index

        Returns:
            Best action index
        """
        return int(np.argmax(self.q_table[state]))


class REINFORCELearner(BaseLearner):
    """
    REINFORCE: Monte Carlo Policy Gradient.

    REINFORCE directly optimizes the policy by following the gradient
    of expected return.

    The policy gradient theorem:
        gradient J(theta) = E[sum_t gradient log pi(a_t|s_t) * G_t]

    Where:
        - J(theta): Expected return (what we want to maximize)
        - pi(a|s): Policy (probability of action a in state s)
        - G_t: Return from time t (discounted sum of future rewards)

    Algorithm:
        1. Collect a complete episode
        2. Compute returns for each timestep
        3. Update policy: increase probability of good actions

    Key properties:
        - On-policy: Must use data from current policy
        - High variance: Returns can vary a lot between episodes
        - Works with any differentiable policy

    Variance reduction:
        - Baseline subtraction: Use (G_t - baseline) instead of G_t
        - The baseline is often V(s) or average return

    Example:
        >>> policy = SimpleNeuralPolicy(state_dim=4, n_actions=2)
        >>> learner = REINFORCELearner(policy=policy, gamma=0.99)
        >>>
        >>> # Collect episode
        >>> states, actions, rewards = [], [], []
        >>> state = env.reset()
        >>> done = False
        >>> while not done:
        ...     action = policy.select_action(state)
        ...     next_state, reward, done = env.step(action)
        ...     states.append(state)
        ...     actions.append(action)
        ...     rewards.append(reward)
        ...     state = next_state
        >>>
        >>> # Train on episode
        >>> learner.train_episode(states, actions, rewards)
    """

    def __init__(
        self,
        policy: SimpleNeuralPolicy,
        gamma: float = 0.99,
        baseline: Optional[str] = 'mean'
    ):
        """
        Initialize REINFORCE learner.

        Args:
            policy: Neural network policy
            gamma: Discount factor
            baseline: Baseline type ('mean', 'none', or custom)
        """
        super().__init__(policy)
        self.gamma = gamma
        self.baseline = baseline

        # Track returns for baseline
        self.return_history: List[float] = []

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        REINFORCE doesn't do per-step updates.
        Use train_episode() instead.
        """
        raise NotImplementedError(
            "REINFORCE requires complete episodes. Use train_episode()."
        )

    def train_episode(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Train on a complete episode.

        Steps:
        1. Compute returns G_t for each timestep
        2. Optionally subtract baseline
        3. Update policy using policy gradient

        Args:
            states: List of states encountered
            actions: List of actions taken
            rewards: List of rewards received

        Returns:
            Dictionary with training metrics
        """
        # Compute returns (discounted cumulative rewards)
        returns = compute_returns(rewards, self.gamma)

        # Track for baseline
        episode_return = returns[0] if returns else 0.0
        self.return_history.append(episode_return)

        # Compute baseline
        if self.baseline == 'mean' and len(self.return_history) > 1:
            baseline_value = np.mean(self.return_history)
        else:
            baseline_value = 0.0

        # Policy gradient update
        total_loss = 0.0
        for t, (state, action, G_t) in enumerate(zip(states, actions, returns)):
            # Advantage = return - baseline
            advantage = G_t - baseline_value

            # Update policy: gradient log pi(a|s) * advantage
            # The policy.update() method handles this
            if isinstance(self.policy, SimpleNeuralPolicy):
                self.policy.update(state, action, advantage)

            # Track loss (negative log prob weighted by advantage)
            log_prob = self.policy.get_log_prob(state, action)
            total_loss += -log_prob * advantage

        avg_loss = total_loss / len(states) if states else 0.0

        return {
            'loss': avg_loss,
            'episode_return': episode_return,
            'baseline': baseline_value
        }


class SimplePPO(BaseLearner):
    """
    Proximal Policy Optimization (PPO) - Simplified Version.

    PPO is a popular policy gradient algorithm that prevents too-large
    policy updates, which can destabilize training.

    The key idea: Clip the policy ratio to prevent large updates

    L_CLIP = E[min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)]

    Where:
        - r(theta) = pi_new(a|s) / pi_old(a|s) (policy ratio)
        - A = advantage (how much better was this action than expected)
        - eps = clipping parameter (typically 0.2)

    Why clipping?
        - Vanilla policy gradient can make huge updates that break the policy
        - Clipping limits how much the policy can change
        - More stable than TRPO (Trust Region Policy Optimization)

    Algorithm:
        1. Collect batch of trajectories with current policy
        2. Compute advantages
        3. For multiple epochs:
           a. Compute policy ratio
           b. Compute clipped surrogate objective
           c. Update policy

    This is a simplified version for educational purposes.
    Production PPO would include value function learning, GAE, etc.

    Example:
        >>> policy = SimpleNeuralPolicy(state_dim=4, n_actions=2)
        >>> ppo = SimplePPO(policy=policy, clip_epsilon=0.2)
        >>>
        >>> # Collect trajectories
        >>> trajectories = collect_trajectories(env, policy, n_episodes=10)
        >>>
        >>> # Update policy
        >>> ppo.train_batch(trajectories)
    """

    def __init__(
        self,
        policy: SimpleNeuralPolicy,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        n_epochs: int = 4,
        value_network: Optional[np.ndarray] = None
    ):
        """
        Initialize PPO learner.

        Args:
            policy: Neural network policy
            gamma: Discount factor
            clip_epsilon: Clipping parameter for policy ratio
            n_epochs: Number of update epochs per batch
            value_network: Optional value function for advantage estimation
        """
        super().__init__(policy)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs

        # Simple value estimates (per-state average return)
        self._value_estimates: Dict[str, float] = {}

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        PPO doesn't do per-step updates.
        Use train_batch() instead.
        """
        raise NotImplementedError(
            "PPO requires batches of trajectories. Use train_batch()."
        )

    def train_episode(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float]
    ) -> Dict[str, float]:
        """
        Train on a single episode (wrapper for train_batch).

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards

        Returns:
            Training metrics
        """
        # Create a single-episode batch
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
        return self.train_batch([trajectory])

    def train_batch(
        self,
        trajectories: List[Dict]
    ) -> Dict[str, float]:
        """
        Train on a batch of trajectories.

        Args:
            trajectories: List of trajectory dicts, each with:
                - 'states': List of states
                - 'actions': List of actions
                - 'rewards': List of rewards

        Returns:
            Training metrics
        """
        # Flatten all trajectories
        all_states = []
        all_actions = []
        all_advantages = []
        all_old_log_probs = []

        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            rewards = traj['rewards']

            # Compute returns and advantages
            returns = compute_returns(rewards, self.gamma)

            # Simple baseline: mean return
            baseline = np.mean(returns) if returns else 0.0
            advantages = [r - baseline for r in returns]

            # Store old log probabilities (before update)
            old_log_probs = []
            for state, action in zip(states, actions):
                log_prob = self.policy.get_log_prob(state, action)
                old_log_probs.append(log_prob)

            all_states.extend(states)
            all_actions.extend(actions)
            all_advantages.extend(advantages)
            all_old_log_probs.extend(old_log_probs)

        # Normalize advantages (helps with training stability)
        all_advantages = np.array(all_advantages)
        if len(all_advantages) > 1:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Multiple epochs of updates
        total_loss = 0.0
        total_clipped = 0

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_clipped = 0

            for i, (state, action, advantage, old_log_prob) in enumerate(
                zip(all_states, all_actions, all_advantages, all_old_log_probs)
            ):
                # Compute current log probability
                new_log_prob = self.policy.get_log_prob(state, action)

                # Policy ratio: pi_new / pi_old = exp(log_new - log_old)
                ratio = np.exp(new_log_prob - old_log_prob)

                # Clipped ratio
                clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                # PPO objective: min(ratio * A, clipped_ratio * A)
                obj1 = ratio * advantage
                obj2 = clipped_ratio * advantage

                # Take minimum (pessimistic bound)
                if advantage >= 0:
                    # When advantage positive, we want to increase probability
                    # But not by more than (1 + epsilon)
                    ppo_objective = min(obj1, obj2)
                else:
                    # When advantage negative, we want to decrease probability
                    # But not by more than (1 - epsilon)
                    ppo_objective = max(obj1, obj2)

                # Check if clipping was active
                if abs(ratio - clipped_ratio) > 1e-6:
                    n_clipped += 1

                # Update policy to maximize objective
                # (policy.update expects advantage, which we've already computed)
                self.policy.update(state, action, ppo_objective)

                epoch_loss += -ppo_objective  # We minimize negative objective

            total_loss += epoch_loss
            total_clipped += n_clipped

        n_samples = len(all_states)
        avg_loss = total_loss / (self.n_epochs * n_samples) if n_samples > 0 else 0.0
        clip_fraction = total_clipped / (self.n_epochs * n_samples) if n_samples > 0 else 0.0

        return {
            'loss': avg_loss,
            'clip_fraction': clip_fraction,
            'n_samples': n_samples
        }


class SARSALearner(BaseLearner):
    """
    SARSA: State-Action-Reward-State-Action Learning.

    SARSA is similar to Q-Learning but is ON-POLICY:
    it learns the value of the policy it's actually following.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))

    Note: Uses Q(s',a') where a' is the ACTUAL next action,
    not max_a' Q(s',a') like Q-Learning.

    Key difference from Q-Learning:
        - Q-Learning (off-policy): Learns optimal Q regardless of exploration
        - SARSA (on-policy): Learns Q for the policy being followed

    Implications:
        - SARSA is more conservative (accounts for exploration in values)
        - Q-Learning can learn faster but may be overoptimistic
        - SARSA is safer in some real-world applications

    Example:
        >>> sarsa = SARSALearner(
        ...     policy=EpsilonGreedyPolicy(n_actions=4),
        ...     n_states=100,
        ...     n_actions=4
        ... )
        >>>
        >>> state = env.reset()
        >>> action = policy.select_action(state)
        >>> while not done:
        ...     next_state, reward, done = env.step(action)
        ...     next_action = policy.select_action(next_state)
        ...     sarsa.update(state, action, reward, next_state, next_action, done)
        ...     state, action = next_state, next_action
    """

    def __init__(
        self,
        policy: EpsilonGreedyPolicy,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99
    ):
        """
        Initialize SARSA learner.

        Args:
            policy: Policy for action selection
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
        """
        super().__init__(policy)
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))

        # Share with policy
        if hasattr(policy, 'q_values'):
            policy.q_values = self.q_table

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform SARSA update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            next_action: Next action (required for SARSA)

        Returns:
            Dictionary with 'td_error'
        """
        current_q = self.q_table[state, action]

        if done:
            td_target = reward
        else:
            # Key difference: use actual next action, not max
            if next_action is None:
                next_action = self.policy.select_action(next_state)
            td_target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = td_target - current_q
        self.q_table[state, action] += self.learning_rate * td_error

        return {'td_error': abs(td_error)}

