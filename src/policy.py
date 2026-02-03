"""
Policy Implementations for Reinforcement Learning
=================================================

A policy defines how an agent selects actions. In RL terminology:
    - Policy (pi): A mapping from states to actions (or action probabilities)
    - Deterministic policy: pi(s) -> a (returns a single action)
    - Stochastic policy: pi(a|s) -> P(a|s) (returns probability distribution)

This module provides several policy implementations:
    1. RandomPolicy: Uniform random action selection (baseline)
    2. EpsilonGreedyPolicy: Epsilon-greedy exploration strategy
    3. SoftmaxPolicy: Temperature-based action selection
    4. SimpleNeuralPolicy: Basic neural network policy

The Exploration-Exploitation Tradeoff:
    A key challenge in RL is balancing:
    - Exploration: Trying new actions to discover potentially better strategies
    - Exploitation: Using the best-known action to maximize immediate reward

    Different policies handle this tradeoff in different ways:
    - Epsilon-greedy: With probability epsilon, explore randomly
    - Softmax: Higher temperature = more exploration
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np


class BasePolicy(ABC):
    """
    Abstract base class for all policies.

    A policy must implement:
        - select_action(): Choose an action given a state
        - get_action_probabilities(): Get probability distribution over actions

    Attributes:
        n_actions (int): Number of possible actions

    Example:
        >>> class MyPolicy(BasePolicy):
        ...     def select_action(self, state):
        ...         # Custom action selection logic
        ...         return 0
        ...     def get_action_probabilities(self, state):
        ...         return np.ones(self.n_actions) / self.n_actions
    """

    def __init__(self, n_actions: int):
        """
        Initialize the policy.

        Args:
            n_actions: Number of possible actions in the action space
        """
        self.n_actions = n_actions

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action given the current state.

        Args:
            state: Current state observation

        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get the probability distribution over actions for a given state.

        Args:
            state: Current state observation

        Returns:
            Array of action probabilities (sums to 1)
        """
        pass

    def update(self, *args, **kwargs) -> None:
        """
        Update the policy parameters (optional).

        Override this method for learnable policies.
        """
        pass


class RandomPolicy(BasePolicy):
    """
    Random policy that selects actions uniformly at random.

    This serves as a baseline for comparison. A good policy should
    significantly outperform random action selection.

    The random policy is useful for:
        - Initial exploration in new environments
        - Baseline comparison for other policies
        - Monte Carlo sampling approaches

    Example:
        >>> policy = RandomPolicy(n_actions=4)
        >>> state = np.array([0.5, 0.3])
        >>> action = policy.select_action(state)  # Random 0-3
        >>> probs = policy.get_action_probabilities(state)  # [0.25, 0.25, 0.25, 0.25]
    """

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        """
        Initialize random policy.

        Args:
            n_actions: Number of possible actions
            seed: Random seed for reproducibility
        """
        super().__init__(n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select a random action (ignores state).

        Args:
            state: Current state (ignored)

        Returns:
            Random action index
        """
        return self.rng.integers(0, self.n_actions)

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Return uniform probability distribution.

        Args:
            state: Current state (ignored)

        Returns:
            Uniform distribution over actions
        """
        return np.ones(self.n_actions) / self.n_actions


class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-greedy policy for balancing exploration and exploitation.

    How it works:
        - With probability (1-epsilon): Select the best action (exploitation)
        - With probability epsilon: Select a random action (exploration)

    Key concepts:
        - Q-values: Estimated value of taking action 'a' in state 's'
        - Greedy action: argmax_a Q(s, a) - the action with highest estimated value
        - Epsilon: Controls exploration rate (typically 0.1 to 0.3)

    Epsilon decay:
        Over time, we often decrease epsilon to explore less as we learn more.
        Common decay strategies: linear decay, exponential decay

    Example:
        >>> policy = EpsilonGreedyPolicy(n_actions=4, epsilon=0.1)
        >>> # Initialize Q-values (4 actions, 10 states)
        >>> policy.q_values = np.random.randn(10, 4)
        >>>
        >>> state_idx = 3
        >>> action = policy.select_action(state_idx)  # 90% greedy, 10% random
        >>>
        >>> # Decay epsilon over time
        >>> policy.decay_epsilon(min_epsilon=0.01)
    """

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize epsilon-greedy policy.

        Args:
            n_actions: Number of possible actions
            epsilon: Initial exploration rate (0 to 1)
            epsilon_decay: Multiplicative decay factor for epsilon
            seed: Random seed for reproducibility
        """
        super().__init__(n_actions)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-values table: will be set by the learning algorithm
        # Shape: (n_states, n_actions) for tabular case
        # or updated externally for function approximation
        self.q_values: Optional[np.ndarray] = None

    def select_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Select action using epsilon-greedy strategy.

        Args:
            state: Current state (index for tabular, array for func approx)

        Returns:
            Selected action index
        """
        # Explore: random action
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)

        # Exploit: best action based on Q-values
        if self.q_values is not None:
            # Tabular case: state is an index
            if isinstance(state, (int, np.integer)):
                return int(np.argmax(self.q_values[state]))
            # Function approximation: state is a feature vector
            # (Would need additional logic for this case)

        # Fallback to random if no Q-values
        return self.rng.integers(0, self.n_actions)

    def get_action_probabilities(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get action probabilities under epsilon-greedy.

        The greedy action gets probability (1-epsilon) + epsilon/n_actions
        Other actions get probability epsilon/n_actions

        Args:
            state: Current state

        Returns:
            Action probability distribution
        """
        # Base probability for all actions (exploration)
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)

        if self.q_values is not None and isinstance(state, (int, np.integer)):
            # Add (1-epsilon) to the greedy action
            greedy_action = np.argmax(self.q_values[state])
            probs[greedy_action] += (1 - self.epsilon)
        else:
            # Uniform if no Q-values
            probs = np.ones(self.n_actions) / self.n_actions

        return probs

    def decay_epsilon(self, min_epsilon: float = 0.01) -> None:
        """
        Decay epsilon to reduce exploration over time.

        Args:
            min_epsilon: Minimum epsilon value to maintain
        """
        self.epsilon = max(min_epsilon, self.epsilon * self.epsilon_decay)


class SoftmaxPolicy(BasePolicy):
    """
    Softmax (Boltzmann) policy for probabilistic action selection.

    How it works:
        P(a|s) = exp(Q(s,a) / temperature) / sum_a' exp(Q(s,a') / temperature)

    Key concepts:
        - Temperature (tau): Controls exploration/exploitation
            - High temperature (tau >> 1): More uniform, more exploration
            - Low temperature (tau << 1): More peaked, more exploitation
            - tau -> 0: Becomes greedy
            - tau -> inf: Becomes uniform random

    Advantages over epsilon-greedy:
        - Actions are chosen proportionally to their estimated value
        - Avoids the "cliff" between greedy and random
        - More nuanced exploration

    Example:
        >>> policy = SoftmaxPolicy(n_actions=4, temperature=1.0)
        >>> policy.q_values = np.array([1.0, 2.0, 0.5, 1.5])  # For one state
        >>> probs = policy.get_action_probabilities(state=0)
        >>> # Higher Q-values get higher probabilities
    """

    def __init__(
        self,
        n_actions: int,
        temperature: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize softmax policy.

        Args:
            n_actions: Number of possible actions
            temperature: Softmax temperature (higher = more exploration)
            seed: Random seed for reproducibility
        """
        super().__init__(n_actions)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.q_values: Optional[np.ndarray] = None

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities with numerical stability.

        We subtract the max value before exp() to prevent overflow:
            softmax(x) = softmax(x - max(x))

        Args:
            values: Array of values to convert to probabilities

        Returns:
            Probability distribution (sums to 1)
        """
        # Scale by temperature
        scaled = values / self.temperature

        # Subtract max for numerical stability (prevents exp overflow)
        scaled = scaled - np.max(scaled)

        # Compute softmax
        exp_values = np.exp(scaled)
        return exp_values / np.sum(exp_values)

    def select_action(self, state: Union[int, np.ndarray]) -> int:
        """
        Select action by sampling from softmax distribution.

        Args:
            state: Current state

        Returns:
            Sampled action index
        """
        probs = self.get_action_probabilities(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def get_action_probabilities(self, state: Union[int, np.ndarray]) -> np.ndarray:
        """
        Get softmax probability distribution over actions.

        Args:
            state: Current state

        Returns:
            Softmax probability distribution
        """
        if self.q_values is not None and isinstance(state, (int, np.integer)):
            return self._softmax(self.q_values[state])

        # Uniform if no Q-values
        return np.ones(self.n_actions) / self.n_actions

    def set_temperature(self, temperature: float) -> None:
        """
        Update the temperature parameter.

        Args:
            temperature: New temperature value
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature


class SimpleNeuralPolicy(BasePolicy):
    """
    Simple neural network policy using NumPy (no deep learning framework).

    Architecture:
        Input (state) -> Hidden Layer -> ReLU -> Output Layer -> Softmax

    This is a basic implementation for educational purposes. For production,
    use PyTorch or TensorFlow.

    Key concepts:
        - Policy gradient: Directly parameterize the policy as a neural network
        - Forward pass: Compute action probabilities from state
        - Backward pass: Update weights to increase probability of good actions

    The policy outputs a probability distribution over actions, which makes
    it naturally stochastic and suitable for policy gradient methods.

    Example:
        >>> # Create policy for 4-dimensional state, 2 actions
        >>> policy = SimpleNeuralPolicy(
        ...     state_dim=4,
        ...     n_actions=2,
        ...     hidden_size=32
        ... )
        >>>
        >>> state = np.array([0.1, 0.2, 0.3, 0.4])
        >>> action = policy.select_action(state)
        >>> probs = policy.get_action_probabilities(state)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_size: int = 32,
        learning_rate: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize neural network policy.

        Args:
            state_dim: Dimension of state input
            n_actions: Number of possible actions
            hidden_size: Number of neurons in hidden layer
            learning_rate: Learning rate for gradient updates
            seed: Random seed for reproducibility
        """
        super().__init__(n_actions)
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Initialize weights using Xavier initialization
        # This helps with gradient flow in deep networks
        self._init_weights()

        # Store intermediate values for backpropagation
        self._cache: dict = {}

    def _init_weights(self) -> None:
        """
        Initialize network weights using Xavier initialization.

        Xavier initialization helps maintain variance of activations
        across layers, leading to better gradient flow.
        """
        # Input -> Hidden weights
        scale1 = np.sqrt(2.0 / (self.state_dim + self.hidden_size))
        self.W1 = self.rng.normal(0, scale1, (self.state_dim, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)

        # Hidden -> Output weights
        scale2 = np.sqrt(2.0 / (self.hidden_size + self.n_actions))
        self.W2 = self.rng.normal(0, scale2, (self.hidden_size, self.n_actions))
        self.b2 = np.zeros(self.n_actions)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function: max(0, x)

        ReLU is popular because:
        - Simple to compute
        - Doesn't saturate for positive values
        - Helps with gradient flow
        """
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: 1 if x > 0, else 0
        """
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax function for converting logits to probabilities.
        """
        # Numerical stability: subtract max
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            state: Input state

        Returns:
            Action probabilities
        """
        # Ensure state is 1D
        state = np.asarray(state).flatten()

        # Layer 1: Linear + ReLU
        z1 = state @ self.W1 + self.b1  # Linear transformation
        a1 = self._relu(z1)              # Activation

        # Layer 2: Linear + Softmax
        z2 = a1 @ self.W2 + self.b2      # Linear transformation
        probs = self._softmax(z2)        # Output probabilities

        # Cache for backward pass
        self._cache = {
            'state': state,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'probs': probs
        }

        return probs

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action by sampling from the policy distribution.

        Args:
            state: Current state

        Returns:
            Sampled action index
        """
        probs = self.forward(state)
        return int(self.rng.choice(self.n_actions, p=probs))

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probability distribution.

        Args:
            state: Current state

        Returns:
            Action probabilities
        """
        return self.forward(state)

    def update(
        self,
        state: np.ndarray,
        action: int,
        advantage: float
    ) -> None:
        """
        Update policy parameters using policy gradient.

        The policy gradient theorem tells us:
            gradient = advantage * gradient_log_pi(a|s)

        This increases probability of actions with positive advantage
        and decreases probability of actions with negative advantage.

        Args:
            state: State where action was taken
            action: Action that was taken
            advantage: Advantage value (how much better than expected)
        """
        # Forward pass to populate cache
        probs = self.forward(state)

        # Compute gradient of log probability
        # d/d_logits log(softmax(logits)[action]) = one_hot(action) - softmax(logits)
        grad_z2 = probs.copy()
        grad_z2[action] -= 1  # This is -(one_hot - probs) = probs - one_hot
        grad_z2 = -grad_z2    # Now it's one_hot - probs

        # Scale by advantage (policy gradient)
        grad_z2 *= advantage

        # Backpropagate to layer 2 weights
        a1 = self._cache['a1']
        grad_W2 = np.outer(a1, grad_z2)
        grad_b2 = grad_z2

        # Backpropagate through ReLU
        grad_a1 = grad_z2 @ self.W2.T
        grad_z1 = grad_a1 * self._relu_derivative(self._cache['z1'])

        # Backpropagate to layer 1 weights
        state = self._cache['state']
        grad_W1 = np.outer(state, grad_z1)
        grad_b1 = grad_z1

        # Update weights (gradient ascent for policy gradient)
        self.W2 += self.learning_rate * grad_W2
        self.b2 += self.learning_rate * grad_b2
        self.W1 += self.learning_rate * grad_W1
        self.b1 += self.learning_rate * grad_b1

    def get_log_prob(self, state: np.ndarray, action: int) -> float:
        """
        Get log probability of taking action in state.

        Log probabilities are used in policy gradient because:
        - More numerically stable than raw probabilities
        - Gradient of log(pi) has nice form for policy gradient

        Args:
            state: State
            action: Action index

        Returns:
            Log probability of action
        """
        probs = self.forward(state)
        return float(np.log(probs[action] + 1e-10))  # Small epsilon for stability

