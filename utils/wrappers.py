"""
Custom Gymnasium wrappers for augmented observations and reward shaping.
"""

import gymnasium as gym
import numpy as np


class EnergyAugmentWrapper(gym.ObservationWrapper):
    """
    Wrapper that augments MountainCar observations with kinetic and potential energy.

    Original observation: [position, velocity]
    Augmented observation: [position, velocity, kinetic_energy, potential_energy]

    This provides the agent with explicit access to energy features, which are
    physically meaningful for understanding the MountainCar dynamics.
    """

    def __init__(self, env):
        """
        Initialize wrapper.

        Args:
            env: Gymnasium environment (MountainCar-v0)
        """
        super().__init__(env)

        # Update observation space to include two extra features
        original_low = self.observation_space.low
        original_high = self.observation_space.high

        # Extend with energy bounds: [0, max_kinetic] and [min_potential, max_potential]
        low = np.concatenate([original_low, [0.0, -1.0]])
        high = np.concatenate([original_high, [1.0, 1.0]])

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # Physical constants from MountainCar
        self.mass = 1.0
        self.gravity = 9.81
        self.env_gravity = 0.0025  # From gymnasium source
        self.max_position = 0.6
        self.min_position = -1.2
        self.goal_position = 0.5

    def observation(self, obs):
        """
        Transform observation by adding kinetic and potential energy features.

        Args:
            obs: Original observation [position, velocity]

        Returns:
            Augmented observation [position, velocity, kinetic, potential]
        """
        pos, vel = obs

        # Kinetic energy: 0.5 * m * v^2
        kinetic = 0.5 * self.mass * (vel ** 2)

        # Potential energy: m * g * h (height = position relative to min)
        # Normalized: position ranges from -1.2 to 0.6, so shift by 1.2
        height = pos - self.min_position
        potential = self.mass * self.env_gravity * height

        # Normalize both to roughly [-1, 1] for numerical stability
        kinetic_normalized = np.clip(kinetic / 1.0, 0.0, 1.0)
        potential_normalized = (potential - 0.02) / 0.02  # Approx range

        return np.array(
            [pos, vel, kinetic_normalized, potential_normalized], dtype=np.float32
        )


class RewardShapingWrapper(gym.RewardWrapper):
    """
    Wrapper that applies reward shaping based on potential functions.

    This wrapper implements potential-based reward shaping, which is theoretically
    guaranteed not to change the optimal policy while helping learning converge faster.
    """

    def __init__(self, env, potential_fn=None, shaping_scale=100.0):
        """
        Initialize wrapper.

        Args:
            env: Gymnasium environment
            potential_fn: Function that computes potential phi(state)
            shaping_scale: Scale factor for shaped bonus
        """
        super().__init__(env)
        self.potential_fn = potential_fn or self._default_potential
        self.shaping_scale = shaping_scale
        self.last_obs = None

    def _default_potential(self, obs):
        """Default potential: position-based (incentivizes moving right)."""
        return obs[0]  # position as potential

    def reset(self, **kwargs):
        """Reset and track first observation."""
        obs, info = super().reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def reward(self, reward):
        """
        Apply potential-based shaping to reward.

        shaped_reward = reward + gamma * phi(s') - phi(s)

        Args:
            reward: Original reward

        Returns:
            Shaped reward
        """
        if self.last_obs is None:
            return reward

        obs = self.env.unwrapped.state  # Current state
        phi_current = self.potential_fn(self.last_obs)
        phi_next = self.potential_fn(np.array(obs))

        # Compute shaping bonus
        gamma = getattr(self.env.unwrapped, "gamma", 0.99)
        shaping_bonus = self.shaping_scale * (gamma * phi_next - phi_current)

        self.last_obs = np.array(obs)

        return reward + shaping_bonus


class CombinedAugmentationWrapper(gym.ObservationWrapper):
    """
    Combines energy augmentation with optional reward shaping.

    This is a convenience wrapper that applies both transformations.
    """

    def __init__(self, env, add_energy=True):
        """
        Initialize combined wrapper.

        Args:
            env: Gymnasium environment
            add_energy: Whether to augment observations with energy features
        """
        super().__init__(env)
        self.add_energy = add_energy

        if add_energy:
            # Update observation space for energy features
            original_low = self.observation_space.low
            original_high = self.observation_space.high
            low = np.concatenate([original_low, [0.0, -1.0]])
            high = np.concatenate([original_high, [1.0, 1.0]])
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )

            self.mass = 1.0
            self.env_gravity = 0.0025
            self.min_position = -1.2

    def observation(self, obs):
        """Transform observation."""
        if not self.add_energy:
            return obs

        pos, vel = obs
        kinetic = 0.5 * self.mass * (vel ** 2)
        height = pos - self.min_position
        potential = self.mass * self.env_gravity * height

        kinetic_normalized = np.clip(kinetic / 1.0, 0.0, 1.0)
        potential_normalized = (potential - 0.02) / 0.02

        return np.array(
            [pos, vel, kinetic_normalized, potential_normalized], dtype=np.float32
        )
