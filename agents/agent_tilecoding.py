"""
Tile Coding with Linear Function Approximation for MountainCar.

This agent uses tile coding to create feature representations of the continuous
state space, then applies linear Q-learning on top of these features.
"""

import numpy as np


class TileCoder:
    """Tile coding feature encoder for continuous state spaces."""

    def __init__(self, low, high, num_tilings=8, tiles_per_dim=None):
        """
        Initialize tile coder.

        Args:
            low: Lower bounds of state space
            high: Upper bounds of state space
            num_tilings: Number of tilings (parallel grids, offset from each other)
            tiles_per_dim: Number of tiles per dimension (default: [8, 8] for 2D)
        """
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.num_tilings = num_tilings
        self.tiles_per_dim = np.array(tiles_per_dim or [8, 8], dtype=int)
        self.state_dim = len(self.low)

        # Precompute tile width for each dimension and tiling
        self.tile_size = (self.high - self.low) / self.tiles_per_dim

        # Offset each tiling by a fraction of tile size
        self.tiling_offsets = np.linspace(0, 1, num_tilings, endpoint=False)

    def encode(self, state):
        """
        Encode state as list of active tile indices.

        Args:
            state: Continuous state vector

        Returns:
            List of tuples, each representing one active tile across all tilings
        """
        state = np.array(state, dtype=float)
        active_tiles = []

        for tiling_idx in range(self.num_tilings):
            # Offset this tiling
            offset = self.tiling_offsets[tiling_idx] * self.tile_size
            adjusted_state = state - self.low + offset

            # Compute tile indices for this tiling
            tile_indices = np.floor(adjusted_state / self.tile_size).astype(int)
            tile_indices = np.clip(tile_indices, 0, self.tiles_per_dim - 1)

            # Create tuple: (tiling_id, pos_tile, vel_tile)
            active_tiles.append(tuple([tiling_idx] + list(tile_indices)))

        return active_tiles

    @property
    def feature_size(self):
        """Total number of features (tiles)."""
        return self.num_tilings * np.prod(self.tiles_per_dim)


class TileCodingQAgent:
    """Q-learning agent using tile coding for continuous state spaces."""

    def __init__(
        self,
        state_low,
        state_high,
        num_actions,
        num_tilings=8,
        tiles_per_dim=None,
        alpha=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
    ):
        """
        Initialize tile coding Q-learning agent.

        Args:
            state_low: Lower bounds of state space
            state_high: Upper bounds of state space
            num_actions: Number of discrete actions
            num_tilings: Number of tilings for tile coding
            tiles_per_dim: Tiles per dimension (default: [8, 8])
            alpha: Learning rate (will be divided by num_tilings for averaging)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay factor for epsilon
        """
        self.tc = TileCoder(state_low, state_high, num_tilings, tiles_per_dim)
        self.num_actions = num_actions
        self.alpha = alpha / self.tc.num_tilings  # Scale by num tilings
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Weight matrix: indexed by tile coordinates -> action values
        # Shape: (num_tilings, tiles_per_dim[0], tiles_per_dim[1], num_actions)
        weight_shape = (
            self.tc.num_tilings,
            *self.tc.tiles_per_dim,
            num_actions,
        )
        self.weights = np.zeros(weight_shape)

    def get_q_values(self, state):
        """
        Compute Q-values for all actions at given state.

        Args:
            state: Continuous state vector

        Returns:
            Array of Q-values, one per action
        """
        tiles = self.tc.encode(state)
        q_values = np.zeros(self.num_actions)

        for tile in tiles:
            q_values += self.weights[tile]

        return q_values

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Continuous state vector
            training: If False, use greedy policy

        Returns:
            Integer action
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using TD learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        tiles = self.tc.encode(state)
        q_current = self.get_q_values(state)[action]

        if done:
            q_target = reward
        else:
            q_next = np.max(self.get_q_values(next_state))
            q_target = reward + self.gamma * q_next

        td_error = q_target - q_current

        # Update all active tiles
        for tile in tiles:
            self.weights[tile + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save weights to disk."""
        np.save(path, self.weights)

    def load(self, path):
        """Load weights from disk."""
        self.weights = np.load(path)
