"""
Scenario 1: Discrete MountainCar, Minimum Steps

Environment: MountainCar-v0
Objective: Reach the goal as quickly as possible
Algorithm: DQN
Reward: Default Gym reward (-1 per step)

Outputs saved:
- results/models/scenario1_dqn.zip
- results/metrics/scenario1_dqn.json
- results/plots/scenario1_dqn_training.png
"""

from pathlib import Path
import json
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# CONFIG
# ============================================================

SCENARIO = 1
ALGORITHM = "DQN"
ENV_NAME = "MountainCar-v0"
OBJECTIVE = "minimum_steps"

TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 100
MAX_STEPS = 200

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


class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []

    def _on_step(self):
        infos = self.locals.get("infos")

        if infos is not None:
            for info in infos:
                if "episode" in info:
                    reward = info["episode"]["r"]
                    steps = info["episode"]["l"]

                    self.episode_rewards.append(reward)
                    self.episode_steps.append(steps)

                    success = 1 if steps < MAX_STEPS else 0
                    self.success_history.append(success)

                    if len(self.episode_rewards) % 20 == 0:
                        recent_reward = np.mean(self.episode_rewards[-20:])
                        recent_steps = np.mean(self.episode_steps[-20:])
                        recent_success = np.mean(self.success_history[-20:]) * 100

                        print(
                            f"Episodes: {len(self.episode_rewards):4d} | "
                            f"Avg Reward: {recent_reward:8.2f} | "
                            f"Avg Steps: {recent_steps:7.2f} | "
                            f"Success: {recent_success:6.1f}%"
                        )

        return True


# ============================================================
# TRAINING
# ============================================================

def train_dqn():
    env = gym.make(ENV_NAME)
    env = Monitor(env)

    callback = TrainingCallback()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0005,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
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

def evaluate_dqn(model):
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
            action = int(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if action != 1:
                fuel_cost += 1

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
    model_path = MODELS_DIR / "scenario1_dqn"
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

    metrics_path = METRICS_DIR / "scenario1_dqn.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to {metrics_path}")


def save_training_plot(callback):
    episode_rewards = callback.episode_rewards
    episode_steps = callback.episode_steps
    success_history = callback.success_history

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

    success_smooth = np.convolve(
        success_history,
        np.ones(window) / window,
        mode="valid"
    ) * 100

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(rewards_smooth)
    plt.title("Scenario 1 DQN Training Performance")
    plt.ylabel("Average Reward")

    plt.subplot(3, 1, 2)
    plt.plot(steps_smooth)
    plt.ylabel("Average Steps")

    plt.subplot(3, 1, 3)
    plt.plot(success_smooth)
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Episode")

    plt.tight_layout()

    plot_path = PLOTS_DIR / "scenario1_dqn_training.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Training plot saved to {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    create_dirs()

    print("=" * 60)
    print("SCENARIO 1: DISCRETE MOUNTAINCAR - MINIMUM STEPS")
    print("ALGORITHM: DQN")
    print("=" * 60)

    model, callback, training_time = train_dqn()

    print("\nEvaluating trained DQN...")
    evaluation_results = evaluate_dqn(model)

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