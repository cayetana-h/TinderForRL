"""
Scenario 3: Discrete MountainCar, Minimum Fuel (Adapted)

Environment: MountainCar-v0 + DiscreteActionCostWrapper (with reward shaping)
Reward: -1 + progress bonus + velocity bonus - fuel cost + goal bonus
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
# DISCRETE ACTION COST WRAPPER (WITH REWARD SHAPING)
# ============================================================================

class DiscreteActionCostWrapper(gym.Wrapper):
    """
    Wrapper with reward shaping for minimum fuel problem.
    
    Shaped reward includes:
    - Base penalty: -1 per step
    - Progress bonus: Rewards moving toward goal
    - Velocity bonus: Encourages building momentum
    - Fuel penalty: Penalizes non-neutral actions
    - Goal bonus: Large reward for reaching the goal
    """

    def __init__(self, env: gym.Env, action_cost=0.001, goal_bonus=100):
        super().__init__(env)
        self.action_cost = action_cost
        self.goal_bonus = goal_bonus
        self.neutral_action = 1
        self.prev_pos = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_pos = obs[0]
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        action_int = int(action)
        pos, vel = obs
        progress = pos - self.prev_pos
        
        # Shaped reward (comparable to Q-learning version)
        reward = -1.0  # Base penalty
        reward += 100.0 * progress  # Strong progress bonus (can be negative)
        reward += 10.0 * abs(vel)   # Velocity bonus
        
        # Penalize fuel usage
        if action_int != self.neutral_action:
            reward -= self.action_cost
        
        # Big bonus for reaching goal
        if terminated:
            reward += self.goal_bonus
        
        self.prev_pos = pos
        
        info = dict(info)
        info["fuel_used"] = 1 if action_int != self.neutral_action else 0
        
        return obs, reward, terminated, truncated, info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class MetricsCallback(BaseCallback):
    """Callback to track episode rewards and steps"""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Check if episode just finished
        if self.locals.get("dones")[0]:
            # Get info from the completed episode
            info = self.locals.get("infos")[0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


# ============================================================================
# TRAINING
# ============================================================================

def train():
    SEED = 42
    TOTAL_TIMESTEPS = 500000
    ACTION_COST = 0.001
    GOAL_BONUS = 100
    
    np.random.seed(SEED)
    
    # Create environment with shaped wrapper
    base_env = gym.make("MountainCar-v0")
    env = DiscreteActionCostWrapper(
        base_env, 
        action_cost=ACTION_COST,
        goal_bonus=GOAL_BONUS
    )
    env = Monitor(env)
    env.reset(seed=SEED)
    
    print("=" * 60)
    print("SCENARIO 3: Discrete MountainCar - Min Fuel (DQN)")
    print("With reward shaping (comparable to Q-learning)")
    print("=" * 60)
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
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
        "scenario": "Scenario 3 - Discrete, Min Fuel (Shaped Rewards)",
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
    print(f"Episodes: {len(rewards)}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Mean reward: {summary['mean_reward']:.2f}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")
    print(f"Results saved to {metrics_dir}")
    print("=" * 60)
    
    # Evaluation
    print("\nEvaluating trained model...")
    eval_env = gym.make("MountainCar-v0")
    eval_env = DiscreteActionCostWrapper(eval_env, action_cost=ACTION_COST, goal_bonus=GOAL_BONUS)
    
    eval_successes = 0
    fuel_used = []
    
    for ep in range(50):
        obs, _ = eval_env.reset()
        fuel = 0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            if int(action) != 1:
                fuel += 1
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            if terminated:
                eval_successes += 1
                break
            if truncated:
                break
        fuel_used.append(fuel)
    
    print(f"Evaluation success rate: {eval_successes / 50:.2%}")
    print(f"Average fuel used: {np.mean(fuel_used):.2f}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()