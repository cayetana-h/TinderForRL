"""
Deep RL training for continuous MountainCar with action intensity cost.

This trains both TD3 and SAC agents on the standard continuous environment
where the reward is shaped by -0.1 * action²(magnitude).

This is the "canonical" deep RL approach for continuous control problems.
"""

import os
import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ActionIntensityCostWrapper(gym.Wrapper):
    """
    Wrapper that adds cost proportional to action magnitude.
    
    reward_shaped = reward_env + (-0.1 * action²)
    
    Standard cost for continuous control where powerful actions are expensive.
    """
    
    def __init__(self, env, intensity_cost=0.1):
        super().__init__(env)
        self.intensity_cost = intensity_cost
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Add cost proportional to action intensity (magnitude squared)
        action_cost = self.intensity_cost * np.sum(action ** 2)
        reward_shaped = reward - action_cost
        
        return obs, reward_shaped, terminated, truncated, info


def train_td3():
    """Train TD3 agent on continuous MountainCar with action intensity cost."""
    print("=" * 70)
    print("Training TD3 (Twin Delayed DDPG)")
    print("=" * 70)
    
    config_path = os.path.join(CURRENT_DIR, "..", "config", "deeprl.yaml")
    config = load_config(config_path)
    
    # Create environment
    env = gym.make("MountainCarContinuous-v0")
    env = ActionIntensityCostWrapper(env, intensity_cost=config["intensity_cost"])
    
    # TD3-specific noise for exploration
    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=config["td3_noise_scale"] * np.ones(env.action_space.shape[0])
    )
    
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=config["td3_learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        policy_delay=config["policy_delay"],
        action_noise=action_noise,
        gamma=config["gamma"],
        verbose=1,
    )
    
    # Train
    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=10,
    )
    
    # Save
    model_dir = os.path.join(CURRENT_DIR, "..", "results", "models")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "td3_continuous_intensity"))
    
    print(f"✓ TD3 model saved to {model_dir}/td3_continuous_intensity")
    
    # Test the policy
    print("\nTesting TD3 policy...")
    test_env = gym.make("MountainCarContinuous-v0")
    test_env = ActionIntensityCostWrapper(test_env, intensity_cost=config["intensity_cost"])
    
    successes = 0
    for episode in range(20):
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                successes += 1
    
    print(f"TD3 Test: {successes}/20 episodes succeeded")
    return model


def train_sac():
    """Train SAC agent on continuous MountainCar with action intensity cost."""
    print("\n" + "=" * 70)
    print("Training SAC (Soft Actor-Critic)")
    print("=" * 70)
    
    config_path = os.path.join(CURRENT_DIR, "..", "config", "deeprl.yaml")
    config = load_config(config_path)
    
    # Create environment
    env = gym.make("MountainCarContinuous-v0")
    env = ActionIntensityCostWrapper(env, intensity_cost=config["intensity_cost"])
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config["sac_learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        gamma=config["gamma"],
        ent_coef="auto",  # Auto-tune entropy coefficient (key advantage of SAC)
        verbose=1,
    )
    
    # Train
    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=10,
    )
    
    # Save
    model_dir = os.path.join(CURRENT_DIR, "..", "results", "models")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "sac_continuous_intensity"))
    
    print(f"✓ SAC model saved to {model_dir}/sac_continuous_intensity")
    
    # Test the policy
    print("\nTesting SAC policy...")
    test_env = gym.make("MountainCarContinuous-v0")
    test_env = ActionIntensityCostWrapper(test_env, intensity_cost=config["intensity_cost"])
    
    successes = 0
    for episode in range(20):
        obs, _ = test_env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                successes += 1
    
    print(f"SAC Test: {successes}/20 episodes succeeded")
    return model


if __name__ == "__main__":
    # Train both models
    td3_model = train_td3()
    sac_model = train_sac()
    
    print("\n" + "=" * 70)
    print("Both models trained and saved successfully!")
    print("=" * 70)
