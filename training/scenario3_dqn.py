"""
Scenario 3: Discrete MountainCar, Minimum Fuel (Adapted)

Environment: MountainCar-v0 + DiscreteActionCostWrapper
Reward: -1 per step + penalty for non-neutral actions (left/right)
Goal: Reach the goal with minimum "fuel" (non-neutral actions)

Algorithm: DQN (Deep Q-Network)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# DISCRETE ACTION COST WRAPPER
# ============================================================================

class DiscreteActionCostWrapper(gym.Wrapper):
    """Wrapper that adds penalty for non-neutral actions"""

    def __init__(self, env: gym.Env, cost_coefficient: float = 0.1):
        super().__init__(env)
        self.cost_coefficient = float(cost_coefficient)
        self.neutral_action = 1

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_int = int(action)
        extra_cost = self.cost_coefficient if action_int != self.neutral_action else 0.0
        shaped_reward = float(reward) - extra_cost
        info = dict(info)
        info["extra_action_cost"] = extra_cost
        return obs, shaped_reward, terminated, truncated, info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        return True


# ============================================================================
# TRAINING
# ============================================================================

def train():
    SEED = 42
    TOTAL_TIMESTEPS = 100000
    COST_COEFFICIENT = 0.1
    
    np.random.seed(SEED)
    
    base_env = gym.make("MountainCar-v0")
    env = DiscreteActionCostWrapper(base_env, cost_coefficient=COST_COEFFICIENT)
    env = Monitor(env)
    env.reset(seed=SEED)
    
    print("=" * 60)
    print("SCENARIO 3: Discrete MountainCar - Min Fuel (DQN)")
    print("=" * 60)
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        seed=SEED,
    )
    
    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    
    results_dir = ensure_dir("results/models")
    metrics_dir = ensure_dir("results/metrics/scenario3_dqn")
    
    model.save(results_dir / "scenario3_dqn")
    
    rewards = np.array(callback.episode_rewards, dtype=np.float32)
    steps = np.array(callback.episode_lengths, dtype=np.int32)
    
    np.save(metrics_dir / "rewards.npy", rewards)
    np.save(metrics_dir / "steps.npy", steps)
    
    successes = steps < 200
    
    summary = {
        "algorithm": "DQN",
        "scenario": "Scenario 3 - Discrete, Min Fuel",
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(np.mean(successes)),
    }
    
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Mean reward: {summary['mean_reward']:.2f}")
    print(f"Results saved to {metrics_dir}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    train()
