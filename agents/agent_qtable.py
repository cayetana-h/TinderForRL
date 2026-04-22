from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class QTableConfig:
    state_low: Sequence[float]
    state_high: Sequence[float]
    num_bins: Sequence[int]
    num_actions: int
    learning_rate: float = 0.2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9998


class QTableAgent:
    def __init__(
        self,
        state_low,
        state_high,
        num_bins,
        num_actions,
        learning_rate=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9998,
    ):
        self.num_bins = np.array(num_bins, dtype=int)
        self.num_actions = int(num_actions)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.state_low = np.array(state_low, dtype=float)
        self.state_high = np.array(state_high, dtype=float)

        self.q_table = np.zeros(tuple(self.num_bins) + (self.num_actions,), dtype=np.float32)
        self.bin_width = np.maximum((self.state_high - self.state_low) / self.num_bins, 1e-12)

    def discretize_state(self, state):
        state = np.asarray(state, dtype=float)
        indices = (state - self.state_low) / self.bin_width
        indices = np.clip(indices.astype(int), 0, self.num_bins - 1)
        return tuple(indices)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state):
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = float(reward) + (0.0 if done else self.gamma * best_next)
        idx = state + (int(action),)
        self.q_table[idx] += self.lr * (target - self.q_table[idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)