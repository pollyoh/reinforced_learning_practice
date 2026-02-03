"""
Environment Implementations for Reinforcement Learning
======================================================

An environment defines the world the agent interacts with.

The standard RL loop:
    1. Agent observes state s
    2. Agent takes action a
    3. Environment returns next state s' and reward r
    4. Repeat until episode ends

Key environment concepts:
    - State space: All possible states
    - Action space: All possible actions
    - Transition dynamics: P(s'|s,a) - probability of next state
    - Reward function: R(s,a,s') - immediate reward

This module provides:
    1. GridWorld: Classic grid navigation (visual, intuitive)
    2. SimpleTextEnvironment: Text-based environment (LLM-relevant)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseEnvironment(ABC):
    """
    Abstract base class for environments.

    Follows a simplified version of the OpenAI Gym interface.

    Required methods:
        - reset(): Start new episode, return initial state
        - step(action): Execute action, return (next_state, reward, done, info)

    Example:
        >>> class MyEnv(BaseEnvironment):
        ...     def reset(self):
        ...         self.state = initial_state
        ...         return self.state
        ...     def step(self, action):
        ...         # Update state based on action
        ...         return next_state, reward, done, info
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward, done, info)
            - next_state: New state after action
            - reward: Immediate reward
            - done: Whether episode has ended
            - info: Additional information (debugging, etc.)
        """
        pass

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of states (for tabular methods)."""
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of possible actions."""
        pass

    def render(self) -> str:
        """
        Render the environment state (optional).

        Returns:
            String representation of current state
        """
        return str(self)


class GridWorld(BaseEnvironment):
    """
    Classic Grid World environment for reinforcement learning.

    The agent navigates a grid to reach a goal while avoiding obstacles.

    Grid layout:
        - 'S': Start position
        - 'G': Goal position
        - '#': Wall/obstacle
        - '.': Empty cell

    Actions:
        - 0: Up
        - 1: Right
        - 2: Down
        - 3: Left

    Rewards:
        - Reaching goal: +10
        - Each step: -0.1 (encourages finding shortest path)
        - Hitting wall: -0.5

    This is a classic RL environment because:
        - Simple enough to understand and debug
        - Complex enough to demonstrate RL concepts
        - Visual/intuitive (easy to see if agent is learning)

    Example:
        >>> env = GridWorld(size=5, goal_position=(4, 4))
        >>> state = env.reset()
        >>> print(env.render())
        # S . . .
        # . . . .
        # . . . .
        # . . . .
        # . . . G

        >>> next_state, reward, done, info = env.step(1)  # Move right
        >>> print(env.render())
        # . S . .
        # . . . .
        # ...
    """

    # Action mappings
    ACTIONS = {
        0: (-1, 0),   # Up
        1: (0, 1),    # Right
        2: (1, 0),    # Down
        3: (0, -1)    # Left
    }
    ACTION_NAMES = ['Up', 'Right', 'Down', 'Left']

    def __init__(
        self,
        size: int = 5,
        start_position: Optional[Tuple[int, int]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_wall: float = -0.5,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize GridWorld environment.

        Args:
            size: Grid size (size x size)
            start_position: Agent starting position (default: (0, 0))
            goal_position: Goal position (default: (size-1, size-1))
            walls: List of wall positions
            reward_goal: Reward for reaching goal
            reward_step: Reward per step (usually negative)
            reward_wall: Reward for hitting wall
            max_steps: Maximum steps per episode
            seed: Random seed
        """
        self.size = size
        self.start_position = start_position or (0, 0)
        self.goal_position = goal_position or (size - 1, size - 1)
        self.walls = set(walls) if walls else set()
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_wall = reward_wall
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Current state
        self.agent_position: Tuple[int, int] = self.start_position
        self.steps_taken: int = 0

    @property
    def n_states(self) -> int:
        """Number of states (grid cells)."""
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        """Number of actions (4 directions)."""
        return 4

    def _position_to_state(self, position: Tuple[int, int]) -> int:
        """Convert (row, col) position to state index."""
        return position[0] * self.size + position[1]

    def _state_to_position(self, state: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        return (state // self.size, state % self.size)

    def reset(self) -> np.ndarray:
        """
        Reset to start of episode.

        Returns:
            Initial state as numpy array [row, col]
        """
        self.agent_position = self.start_position
        self.steps_taken = 0
        return np.array(self.agent_position)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return result.

        Args:
            action: Action index (0-3)

        Returns:
            (next_state, reward, done, info)
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")

        self.steps_taken += 1

        # Calculate new position
        delta = self.ACTIONS[action]
        new_row = self.agent_position[0] + delta[0]
        new_col = self.agent_position[1] + delta[1]
        new_position = (new_row, new_col)

        # Check if move is valid
        info = {'invalid_action': False, 'reached_goal': False}

        # Check boundaries
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            # Hit boundary - stay in place
            reward = self.reward_wall
            info['invalid_action'] = True
            info['hit_boundary'] = True
        elif new_position in self.walls:
            # Hit wall - stay in place
            reward = self.reward_wall
            info['invalid_action'] = True
            info['hit_wall'] = True
        else:
            # Valid move
            self.agent_position = new_position

            if new_position == self.goal_position:
                reward = self.reward_goal
                info['reached_goal'] = True
            else:
                reward = self.reward_step

        # Check termination
        done = (
            self.agent_position == self.goal_position or
            self.steps_taken >= self.max_steps
        )

        info['steps'] = self.steps_taken
        info['position'] = self.agent_position

        return np.array(self.agent_position), reward, done, info

    def render(self) -> str:
        """
        Render the grid as ASCII art.

        Returns:
            String representation of the grid
        """
        lines = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                pos = (row, col)
                if pos == self.agent_position:
                    char = 'A'  # Agent
                elif pos == self.goal_position:
                    char = 'G'  # Goal
                elif pos in self.walls:
                    char = '#'  # Wall
                elif pos == self.start_position:
                    char = 'S'  # Start
                else:
                    char = '.'  # Empty
                line.append(char)
            lines.append(' '.join(line))

        return '\n'.join(lines)

    def get_state_index(self) -> int:
        """
        Get current state as integer index (for tabular methods).

        Returns:
            State index (0 to n_states-1)
        """
        return self._position_to_state(self.agent_position)


class SimpleTextEnvironment(BaseEnvironment):
    """
    Simple text-based environment for demonstrating RL with text.

    This environment simulates a simple text completion task where
    the agent must learn to select appropriate words to complete prompts.

    While this is a simplified version, it demonstrates concepts relevant
    to training language models with RL (RLHF).

    Setup:
        - States: Prompt contexts (encoded as indices)
        - Actions: Word choices
        - Rewards: Based on appropriateness of word choice

    This is educational - real LLM training uses:
        - Continuous state spaces (embeddings)
        - Much larger action spaces (full vocabulary)
        - More sophisticated reward models

    Example:
        >>> env = SimpleTextEnvironment()
        >>> state = env.reset()  # Get a prompt
        >>> print(env.get_prompt())  # "The cat sat on the ___"
        >>>
        >>> action = 2  # Select word index 2
        >>> next_state, reward, done, info = env.step(action)
        >>> print(info['selected_word'])  # "mat" (if that's word 2)
        >>> print(reward)  # High if appropriate, low otherwise
    """

    # Sample prompts and appropriate completions
    DEFAULT_PROMPTS = [
        {
            'text': "The cat sat on the",
            'good_words': ['mat', 'floor', 'couch'],
            'bad_words': ['sky', 'ocean', 'moon']
        },
        {
            'text': "The sun rises in the",
            'good_words': ['east', 'morning', 'sky'],
            'bad_words': ['west', 'night', 'ground']
        },
        {
            'text': "Water freezes at zero degrees",
            'good_words': ['Celsius', 'temperature', 'centigrade'],
            'bad_words': ['Fahrenheit', 'meters', 'miles']
        },
        {
            'text': "Birds fly through the",
            'good_words': ['air', 'sky', 'clouds'],
            'bad_words': ['water', 'ground', 'rock']
        },
        {
            'text': "Fish swim in the",
            'good_words': ['water', 'ocean', 'sea'],
            'bad_words': ['air', 'sky', 'land']
        }
    ]

    def __init__(
        self,
        prompts: Optional[List[Dict]] = None,
        reward_good: float = 1.0,
        reward_bad: float = -1.0,
        reward_neutral: float = 0.0,
        max_steps: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize text environment.

        Args:
            prompts: List of prompt dictionaries with 'text', 'good_words', 'bad_words'
            reward_good: Reward for selecting appropriate word
            reward_bad: Reward for selecting inappropriate word
            reward_neutral: Reward for neutral words
            max_steps: Maximum completions per episode
            seed: Random seed
        """
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.reward_good = reward_good
        self.reward_bad = reward_bad
        self.reward_neutral = reward_neutral
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Build vocabulary (all unique words across prompts)
        self.vocabulary: List[str] = []
        for prompt in self.prompts:
            self.vocabulary.extend(prompt['good_words'])
            self.vocabulary.extend(prompt['bad_words'])
        self.vocabulary = list(set(self.vocabulary))
        self.vocabulary.sort()  # For reproducibility

        # Word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        # Current state
        self.current_prompt_idx: int = 0
        self.steps_taken: int = 0
        self.completions: List[str] = []

    @property
    def n_states(self) -> int:
        """Number of states (number of prompts)."""
        return len(self.prompts)

    @property
    def n_actions(self) -> int:
        """Number of actions (vocabulary size)."""
        return len(self.vocabulary)

    def reset(self) -> np.ndarray:
        """
        Reset to start of episode with random prompt.

        Returns:
            State array (prompt index as one-hot or simple index)
        """
        self.current_prompt_idx = self.rng.integers(0, len(self.prompts))
        self.steps_taken = 0
        self.completions = []

        # Return one-hot encoded prompt index
        state = np.zeros(len(self.prompts))
        state[self.current_prompt_idx] = 1.0
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute word selection action.

        Args:
            action: Index of word in vocabulary

        Returns:
            (next_state, reward, done, info)
        """
        if not (0 <= action < len(self.vocabulary)):
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        # Get selected word
        selected_word = self.vocabulary[action]
        self.completions.append(selected_word)

        # Get current prompt
        prompt = self.prompts[self.current_prompt_idx]

        # Determine reward based on word appropriateness
        info = {
            'selected_word': selected_word,
            'prompt': prompt['text'],
            'is_good': False,
            'is_bad': False
        }

        if selected_word in prompt['good_words']:
            reward = self.reward_good
            info['is_good'] = True
        elif selected_word in prompt['bad_words']:
            reward = self.reward_bad
            info['is_bad'] = True
        else:
            reward = self.reward_neutral

        # Move to next prompt (or end if max_steps reached)
        self.current_prompt_idx = self.rng.integers(0, len(self.prompts))

        done = self.steps_taken >= self.max_steps

        # Next state
        next_state = np.zeros(len(self.prompts))
        next_state[self.current_prompt_idx] = 1.0

        info['steps'] = self.steps_taken
        info['completions'] = self.completions.copy()

        return next_state, reward, done, info

    def get_prompt(self) -> str:
        """Get the current prompt text."""
        return self.prompts[self.current_prompt_idx]['text']

    def get_vocabulary(self) -> List[str]:
        """Get the vocabulary list."""
        return self.vocabulary.copy()

    def render(self) -> str:
        """
        Render current state as text.

        Returns:
            String showing prompt and recent completions
        """
        prompt = self.get_prompt()
        recent = self.completions[-3:] if self.completions else []

        lines = [
            f"Prompt: \"{prompt} ___\"",
            f"Recent completions: {recent}",
            f"Steps: {self.steps_taken}/{self.max_steps}"
        ]

        return '\n'.join(lines)


class MultiArmedBandit(BaseEnvironment):
    """
    Multi-Armed Bandit: Simplest RL setting (no state transitions).

    Imagine a row of slot machines (bandits), each with different
    (unknown) payout probabilities. Your goal: maximize total reward.

    This is a stateless problem - perfect for understanding:
        - Exploration vs exploitation
        - Action-value estimation
        - Epsilon-greedy, UCB, Thompson Sampling

    Why "multi-armed"?
        - Old slot machines had one arm (lever) to pull
        - Multiple machines = multiple arms = multiple actions

    Key insight:
        - No state transitions (always same situation)
        - Pure exploration/exploitation tradeoff
        - Foundation for more complex RL

    Example:
        >>> bandit = MultiArmedBandit(n_arms=10)
        >>>
        >>> # Try arm 3
        >>> _, reward, _, _ = bandit.step(3)
        >>> print(reward)  # Random reward based on arm 3's distribution
    """

    def __init__(
        self,
        n_arms: int = 10,
        reward_means: Optional[np.ndarray] = None,
        reward_stds: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize bandit.

        Args:
            n_arms: Number of arms (actions)
            reward_means: Mean reward for each arm
            reward_stds: Standard deviation of reward for each arm
            seed: Random seed
        """
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

        # Initialize reward distributions
        if reward_means is not None:
            self.reward_means = np.array(reward_means)
        else:
            # Random means between 0 and 1
            self.reward_means = self.rng.random(n_arms)

        if reward_stds is not None:
            self.reward_stds = np.array(reward_stds)
        else:
            # Standard deviation of 0.5 for all arms
            self.reward_stds = np.ones(n_arms) * 0.5

        # Track statistics
        self.pulls = np.zeros(n_arms)
        self.total_reward = 0.0
        self.steps = 0

    @property
    def n_states(self) -> int:
        """Bandits have only 1 state (stateless)."""
        return 1

    @property
    def n_actions(self) -> int:
        """Number of arms."""
        return self.n_arms

    def reset(self) -> np.ndarray:
        """
        Reset statistics (optional for bandits).

        Returns:
            Dummy state (always 0 for bandits)
        """
        self.pulls = np.zeros(self.n_arms)
        self.total_reward = 0.0
        self.steps = 0
        return np.array([0])  # Dummy state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Pull an arm and receive reward.

        Args:
            action: Arm to pull (0 to n_arms-1)

        Returns:
            (state, reward, done, info)
            - state: Dummy (bandits are stateless)
            - reward: Sampled from arm's distribution
            - done: Always False (bandits don't terminate)
            - info: Statistics
        """
        if not (0 <= action < self.n_arms):
            raise ValueError(f"Invalid action: {action}")

        # Sample reward from arm's distribution
        reward = self.rng.normal(
            self.reward_means[action],
            self.reward_stds[action]
        )

        # Update statistics
        self.pulls[action] += 1
        self.total_reward += reward
        self.steps += 1

        info = {
            'arm': action,
            'true_mean': self.reward_means[action],
            'pulls': self.pulls.copy(),
            'total_reward': self.total_reward,
            'optimal_arm': int(np.argmax(self.reward_means)),
            'regret': np.max(self.reward_means) - self.reward_means[action]
        }

        return np.array([0]), reward, False, info

    def render(self) -> str:
        """Render bandit statistics."""
        lines = [
            f"Multi-Armed Bandit ({self.n_arms} arms)",
            f"Steps: {self.steps}",
            f"Total reward: {self.total_reward:.2f}",
            f"Pulls per arm: {self.pulls.astype(int)}",
            f"(Optimal arm: {np.argmax(self.reward_means)})"
        ]
        return '\n'.join(lines)

