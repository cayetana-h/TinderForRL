from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


def safe_mean(values: Sequence[float] | np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def safe_std(values: Sequence[float] | np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


def rolling_mean(values: Sequence[float], window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), arr.size))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="valid")


@dataclass
class EpisodeSummary:
    episodes: int
    success_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    std_steps: float
    mean_cost: float
    std_cost: float
    max_reward: float
    min_reward: float


def summarize_episodes(
    rewards: Sequence[float],
    steps: Sequence[int],
    costs: Sequence[float] | None = None,
    successes: Sequence[bool] | None = None,
) -> dict:
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    steps_arr = np.asarray(steps, dtype=np.float32)
    costs_arr = np.asarray(costs if costs is not None else [0.0] * len(rewards), dtype=np.float32)
    successes_arr = np.asarray(successes if successes is not None else [False] * len(rewards), dtype=bool)

    if rewards_arr.size == 0:
        summary = EpisodeSummary(
            episodes=0,
            success_rate=0.0,
            mean_reward=0.0,
            std_reward=0.0,
            mean_steps=0.0,
            std_steps=0.0,
            mean_cost=0.0,
            std_cost=0.0,
            max_reward=0.0,
            min_reward=0.0,
        )
        return asdict(summary)

    summary = EpisodeSummary(
        episodes=int(rewards_arr.size),
        success_rate=float(successes_arr.mean()) if successes_arr.size else 0.0,
        mean_reward=float(rewards_arr.mean()),
        std_reward=float(rewards_arr.std()),
        mean_steps=float(steps_arr.mean()) if steps_arr.size else 0.0,
        std_steps=float(steps_arr.std()) if steps_arr.size else 0.0,
        mean_cost=float(costs_arr.mean()) if costs_arr.size else 0.0,
        std_cost=float(costs_arr.std()) if costs_arr.size else 0.0,
        max_reward=float(rewards_arr.max()),
        min_reward=float(rewards_arr.min()),
    )
    return asdict(summary)