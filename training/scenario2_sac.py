"""
Scenario 2: Continuous MountainCar, Minimum Fuel (Squared Cost)

Environment: MountainCarContinuous-v0
Reward: -0.1 * action² (built into environment)
Goal: Reach the goal with minimum energy (fuel) usage

Algorithm: SAC (Soft Actor-Critic)
"""

from __future__ import annotations
from pathlib import Path
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import SAC
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
    """Callback to track episode rewards"""
    
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
    # Configuration
    SEED = 42
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 0.0003
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99
    
    np.random.seed(SEED)
    
    # Create environment
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env)
    env.reset(seed=SEED)
    
    print("=" * 60)
    print("SCENARIO 2: Continuous MountainCar - Min Fuel Squared (SAC)")
    print("=" * 60)
    
    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        ent_coef="auto",
        verbose=1,
        seed=SEED,
    )
    
    # Train
    callback = MetricsCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    
    # Save model
    results_dir = ensure_dir("results/models")
    metrics_dir = ensure_dir("results/metrics/scenario2_sac")
    
    model.save(results_dir / "scenario2_sac")
    
    # Save metrics
    rewards = np.array(callback.episode_rewards, dtype=np.float32)
    steps = np.array(callback.episode_lengths, dtype=np.int32)
    
    np.save(metrics_dir / "rewards.npy", rewards)
    np.save(metrics_dir / "steps.npy", steps)
    
    # Calculate success rate (reached goal if reward > 90)
    successes = rewards > 90
    
    # Save summary
    summary = {
        "algorithm": "SAC",
        "scenario": "Scenario 2 - Continuous, Min Fuel (Squared)",
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
