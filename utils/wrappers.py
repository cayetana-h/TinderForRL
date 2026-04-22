from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class DiscreteActionCostWrapper(gym.Wrapper):
    """
    MountainCar-v0 wrapper that subtracts an extra fixed cost whenever the action is not neutral.
    Action 1 is assumed to be neutral in the discrete action space {0,1,2}.
    """

    def __init__(self, env: gym.Env, cost_coefficient: float = 0.1, neutral_action: int = 1):
        super().__init__(env)
        self.cost_coefficient = float(cost_coefficient)
        self.neutral_action = int(neutral_action)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extra_cost = self.cost_coefficient if int(action) != self.neutral_action else 0.0
        reward = float(reward) - extra_cost
        info = dict(info)
        info["extra_action_cost"] = extra_cost
        return obs, reward, terminated, truncated, info


class ContinuousExtraActionCostWrapper(gym.Wrapper):
    """
    Optional extra shaping wrapper for MountainCarContinuous-v0.
    The base environment already uses -0.1 * action^2, so by default use extra_cost_coefficient=0.0.
    """

    def __init__(self, env: gym.Env, extra_cost_coefficient: float = 0.0):
        super().__init__(env)
        self.extra_cost_coefficient = float(extra_cost_coefficient)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_arr = np.asarray(action, dtype=np.float32)
        intensity = float(np.sum(np.square(action_arr)))
        extra_cost = self.extra_cost_coefficient * intensity
        reward = float(reward) - extra_cost
        info = dict(info)
        info["action_intensity"] = intensity
        info["extra_action_cost"] = extra_cost
        return obs, reward, terminated, truncated, info