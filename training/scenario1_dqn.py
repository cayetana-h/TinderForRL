"""
Scenario 1: Discrete MountainCar, Minimum Steps (Standard)

Environment: MountainCar-v0
Reward: -1 per step (standard)
Goal: Reach the goal as quickly as possible

Algorithm: DQN (Deep Q-Network)
"""

from __future__ import annotations
from pathlib import Path
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
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
    # Configuration
    SEED = 42
    TOTAL_TIMESTEPS = 500000
    LEARNING_RATE = 0.001
    BUFFER_SIZE = 50000
    LEARNING_STARTS = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99
    
    np.random.seed(SEED)
    
    # Create environment
    env = gym.make("MountainCar-v0")
    env = Monitor(env)
    env.reset(seed=SEED)
    
    print("=" * 60)
    print("SCENARIO 1: Discrete MountainCar - Minimum Steps (DQN)")
    print("=" * 60)
    
    # Create DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        seed=SEED,
    )
    
    # Train
    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    
    # Save model
    results_dir = ensure_dir("results/models")
    metrics_dir = ensure_dir("results/metrics/scenario1_dqn")
    
    model.save(results_dir / "scenario1_dqn")
    
    # Save metrics
    rewards = np.array(callback.episode_rewards, dtype=np.float32)
    steps = np.array(callback.episode_lengths, dtype=np.int32)
    
    np.save(metrics_dir / "rewards.npy", rewards)
    np.save(metrics_dir / "steps.npy", steps)
    
    # Calculate success rate (reached goal if episode length < 200)
    successes = steps < 200
    
    # Save summary
    summary = {
        "algorithm": "DQN",
        "scenario": "Scenario 1 - Discrete, Min Steps",
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
    
    env.close()


if __name__ == "__main__":
    train()