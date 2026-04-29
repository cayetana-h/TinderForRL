"""
Scenario 2: Continuous MountainCar, Minimum Fuel

Environment: MountainCarContinuous-v0
Objective: Reach the goal while minimizing fuel usage
Algorithm: SAC

Important:
Training uses reward shaping so the agent does not learn to simply stay still.
Evaluation uses the normal environment to report real performance.

Outputs saved:
- results/models/scenario2_sac.zip
- results/metrics/scenario2_sac.json
- results/plots/scenario2_sac_training.png
"""

from pathlib import Path
import json
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# CONFIG
# ============================================================

SCENARIO = 2
ALGORITHM = "SAC"
ENV_NAME = "MountainCarContinuous-v0"
OBJECTIVE = "minimum_fuel"

TOTAL_TIMESTEPS = 300_000
EVAL_EPISODES = 100
MAX_STEPS = 999

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"


# ============================================================
# SETUP
# ============================================================

def create_dirs():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class FuelEfficientMountainCar(gym.Wrapper):
    """
    Stronger reward shaping wrapper for minimum fuel objective.
    Enhanced incentives to help agent discover solution.
    """
    def __init__(self, env):
        super().__init__(env)
        self.previous_position = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.previous_position = state[0]
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        position = next_state[0]
        velocity = next_state[1]

        progress = position - self.previous_position
        fuel_cost = float(action[0] ** 2)

        shaped_reward = reward

        # STRONGER progress bonus - critical for discovering solution
        shaped_reward += 25.0 * progress

        # Reward building momentum
        shaped_reward += 1.0 * abs(velocity)

        # Small fuel penalty (still min-fuel scenario)
        shaped_reward -= 0.01 * fuel_cost

        # Very strong success signal
        if terminated:
            shaped_reward += 200.0

        # Small penalty for being stuck in the valley
        if abs(position) < 0.2 and abs(velocity) < 0.01:
            shaped_reward -= 0.5

        self.previous_position = position

        return next_state, shaped_reward, terminated, truncated, info


class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0

    def _on_step(self):
        infos = self.locals.get("infos")

        if infos is not None:
            for info in infos:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    steps = info["episode"]["l"]

                    self.episode_rewards.append(reward)
                    self.episode_steps.append(steps)
                    
                    if steps < MAX_STEPS:
                        self.success_count += 1

                    if len(self.episode_rewards) % 10 == 0:
                        recent_reward = np.mean(self.episode_rewards[-10:])
                        recent_steps = np.mean(self.episode_steps[-10:])
                        success_rate = (self.success_count / len(self.episode_rewards)) * 100

                        print(
                            f"Episodes: {len(self.episode_rewards):4d} | "
                            f"Avg Training Reward: {recent_reward:8.2f} | "
                            f"Avg Steps: {recent_steps:7.2f} | "
                            f"Success Rate: {success_rate:5.1f}%"
                        )

        return True


# ============================================================
# TRAINING
# ============================================================

def train_sac():
    env = gym.make(ENV_NAME)
    env = FuelEfficientMountainCar(env)
    env = Monitor(env)

    callback = TrainingCallback()

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=300_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",  # Auto-tune entropy
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=42
    )

    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )

    training_time = time.time() - start_time
    env.close()

    return model, callback, training_time


# ============================================================
# EVALUATION
# ============================================================

def evaluate_sac(model):
    # Important: normal environment, not shaped wrapper.
    env = gym.make(ENV_NAME)

    rewards = []
    steps_list = []
    successes = []
    fuel_costs = []

    for episode in range(EVAL_EPISODES):
        state, _ = env.reset()

        total_reward = 0
        steps = 0
        success = False
        fuel_cost = 0

        for step in range(MAX_STEPS):
            action, _ = model.predict(state, deterministic=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            action_value = float(action[0])
            fuel_cost += action_value ** 2

            total_reward += reward
            steps += 1
            state = next_state

            if terminated:
                success = True
                break

            if done:
                break

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(1 if success else 0)
        fuel_costs.append(fuel_cost)

    env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "average_reward": float(np.mean(rewards)),
        "average_steps": float(np.mean(steps_list)),
        "average_fuel": float(np.mean(fuel_costs)),
        "min_steps": int(np.min(steps_list)),
        "max_steps": int(np.max(steps_list))
    }


# ============================================================
# SAVING
# ============================================================

def save_model(model):
    model_path = MODELS_DIR / "scenario2_sac"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")


def save_metrics(evaluation_results, training_time):
    metrics = {
        "scenario": SCENARIO,
        "algorithm": ALGORITHM,
        "environment": ENV_NAME,
        "objective": OBJECTIVE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "evaluation_episodes": EVAL_EPISODES,
        "success_rate": evaluation_results["success_rate"],
        "success_rate_percent": evaluation_results["success_rate"] * 100,
        "average_reward": evaluation_results["average_reward"],
        "average_steps": evaluation_results["average_steps"],
        "average_fuel": evaluation_results["average_fuel"],
        "min_steps": evaluation_results["min_steps"],
        "max_steps": evaluation_results["max_steps"],
        "training_time_seconds": training_time
    }

    metrics_path = METRICS_DIR / "scenario2_sac.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to {metrics_path}")


def save_training_plot(callback):
    episode_rewards = callback.episode_rewards
    episode_steps = callback.episode_steps

    if len(episode_rewards) < 10:
        print("Not enough episode data to create training plot.")
        return

    window = 10

    rewards_smooth = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    steps_smooth = np.convolve(
        episode_steps,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(rewards_smooth)
    plt.title("Scenario 2 SAC Training Performance")
    plt.ylabel("Avg Training Reward")

    plt.subplot(2, 1, 2)
    plt.plot(steps_smooth)
    plt.ylabel("Average Steps")
    plt.xlabel("Episode")

    plt.tight_layout()

    plot_path = PLOTS_DIR / "scenario2_sac_training.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Training plot saved to {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    create_dirs()

    print("=" * 60)
    print("SCENARIO 2: CONTINUOUS MOUNTAINCAR - MINIMUM FUEL")
    print("ALGORITHM: SAC")
    print("=" * 60)
    print(f"Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print("Enhanced Reward Shaping: Enabled")
    print("Auto-tuned Entropy: Enabled")
    print("=" * 60)

    model, callback, training_time = train_sac()

    print("\nEvaluating trained SAC...")
    evaluation_results = evaluate_sac(model)

    print("\nEvaluation Results:")
    print(f"Success Rate: {evaluation_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {evaluation_results['average_reward']:.2f}")
    print(f"Average Steps: {evaluation_results['average_steps']:.2f}")
    print(f"Average Fuel: {evaluation_results['average_fuel']:.2f}")
    print(f"Min Steps: {evaluation_results['min_steps']}")
    print(f"Max Steps: {evaluation_results['max_steps']}")
    print(f"Training Time: {training_time:.2f} seconds")

    save_model(model)
    save_metrics(evaluation_results, training_time)
    save_training_plot(callback)

    print("\nDone.")


if __name__ == "__main__":
    main()