from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class DiscreteActionCostWrapper(gym.Wrapper):
    """
    Wrapper for MountainCar-v0 that adds an extra penalty when the agent takes
    a non-neutral discrete action.

    In MountainCar-v0, actions are:
        0 -> push left
        1 -> no push (neutral)
        2 -> push right

    This wrapper keeps the environment discrete but makes the reward slightly
    more expensive whenever the agent is not staying neutral.
    """

    def __init__(
        self,
        env: gym.Env,
        cost_coefficient: float = 0.1,
        neutral_action: int = 1,
    ):
        super().__init__(env)
        self.cost_coefficient = float(cost_coefficient)
        self.neutral_action = int(neutral_action)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action_int = int(action)
        extra_cost = self.cost_coefficient if action_int != self.neutral_action else 0.0
        shaped_reward = float(reward) - extra_cost

        info = dict(info)
        info["extra_action_cost"] = extra_cost
        info["raw_reward"] = float(reward)

        return obs, shaped_reward, terminated, truncated, info


class ContinuousExtraActionCostWrapper(gym.Wrapper):
    """
    Optional wrapper for MountainCarContinuous-v0.

    The base environment already uses an action-dependent cost internally.
    By default this wrapper does NOT add any extra penalty unless you explicitly
    set extra_cost_coefficient > 0.

    This is useful if you want to test additional reward shaping on top of the
    standard continuous-control formulation.
    """

    def __init__(self, env: gym.Env, extra_cost_coefficient: float = 0.0):
        super().__init__(env)
        self.extra_cost_coefficient = float(extra_cost_coefficient)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action_arr = np.asarray(action, dtype=np.float32)
        intensity = float(np.sum(np.square(action_arr)))
        extra_cost = self.extra_cost_coefficient * intensity
        shaped_reward = float(reward) - extra_cost

        info = dict(info)
        info["action_intensity"] = intensity
        info["extra_action_cost"] = extra_cost
        info["raw_reward"] = float(reward)

        return obs, shaped_reward, terminated, truncated, info