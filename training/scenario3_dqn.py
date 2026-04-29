"""
Scenario 3: Discrete MountainCar, Minimum Fuel

Environment: MountainCar-v0
Objective: Reach the goal while minimizing fuel usage (non-neutral actions)
Algorithm: DQN (Deep Q-Network)
Reward: Shaped reward with progress, velocity, fuel penalty, and goal bonus

Outputs saved:
- results/scenario3/models/scenario3_dqn.zip
- results/scenario3/metrics/scenario3_dqn.json
- results/scenario3/plots/scenario3_dqn_training.png
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

SCENARIO = 3
ALGORITHM = "DQN"
ENV_NAME = "MountainCar-v0"
OBJECTIVE = "minimum_fuel"

TOTAL_TIMESTEPS = 500_000
EVAL_EPISODES = 100

ACTION_COST = 0.001
GOAL_BONUS = 100

MAX_STEPS = 200

SEED = 42

RESULTS_DIR = Path("results") / "scenario3"
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
    def __init__(self, env, action_cost=0.001, goal_bonus=100):
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

        pos, vel = obs
        progress = pos - self.prev_pos

        # Shaped reward (same as Q-learning version)
        reward = -1.0
        reward += 100.0 * progress
        reward += 10.0 * abs(vel)

        if action != self.neutral_action:
            reward -= self.action_cost

        if terminated:
            reward += self.goal_bonus

        self.prev_pos = pos

        return obs, reward, terminated, truncated, info


class TrainingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_fuel = []
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

                    # Approximate fuel usage (we don't track it during training)
                    # This is a rough estimate
                    self.episode_fuel.append(steps * 0.8)  # Placeholder

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
    env = DiscreteActionCostWrapper(env, action_cost=ACTION_COST, goal_bonus=GOAL_BONUS)
    env.action_space.seed(SEED)
    env.reset(seed=SEED)
    env = Monitor(env)

    callback = TrainingCallback()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=SEED
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
    # Evaluate on wrapped environment (same as training)
    env = gym.make(ENV_NAME)
    env = DiscreteActionCostWrapper(env, action_cost=ACTION_COST, goal_bonus=GOAL_BONUS)
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    rewards = []
    steps_list = []
    successes = []
    fuel_costs = []

    for _ in range(EVAL_EPISODES):
        state, _ = env.reset()

        total_reward = 0
        steps = 0
        success = False
        fuel_cost = 0

        for _ in range(MAX_STEPS):
            action, _ = model.predict(state, deterministic=True)
            action = int(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Track fuel (non-neutral actions)
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
    model_path = MODELS_DIR / "scenario3_dqn"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")


def save_metrics(results, training_time):
    metrics = {
        "scenario": SCENARIO,
        "algorithm": ALGORITHM,
        "environment": ENV_NAME,
        "objective": OBJECTIVE,
        "total_timesteps": TOTAL_TIMESTEPS,
        "evaluation_episodes": EVAL_EPISODES,
        "seed": SEED,
        "success_rate": results["success_rate"],
        "success_rate_percent": results["success_rate"] * 100,
        "average_reward": results["average_reward"],
        "average_steps": results["average_steps"],
        "average_fuel": results["average_fuel"],
        "min_steps": results["min_steps"],
        "max_steps": results["max_steps"],
        "training_time_seconds": training_time
    }

    path = METRICS_DIR / "scenario3_dqn.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {path}")


def save_training_plot(callback):
    if len(callback.episode_rewards) < 10:
        return

    window = 10

    rewards = np.convolve(callback.episode_rewards, np.ones(window)/window, mode="valid")
    steps = np.convolve(callback.episode_steps, np.ones(window)/window, mode="valid")
    success = np.convolve(callback.success_history, np.ones(window)/window, mode="valid") * 100

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("Scenario 3 DQN Training Performance")
    plt.ylabel("Reward")

    plt.subplot(3, 1, 2)
    plt.plot(steps)
    plt.ylabel("Steps")

    plt.subplot(3, 1, 3)
    plt.plot(success)
    plt.ylabel("Success %")
    plt.xlabel("Episodes")

    plt.tight_layout()

    path = PLOTS_DIR / "scenario3_dqn_training.png"
    plt.savefig(path)
    plt.close()

    print(f"Training plot saved to {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    create_dirs()

    print("=" * 60)
    print("SCENARIO 3: DISCRETE MOUNTAINCAR - MINIMUM FUEL")
    print("ALGORITHM: DQN")
    print("=" * 60)

    model, callback, training_time = train_dqn()

    print("\nEvaluating trained DQN...")
    results = evaluate_dqn(model)

    print("\nEvaluation Results:")
    print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    print(f"Average Steps: {results['average_steps']:.2f}")
    print(f"Average Fuel: {results['average_fuel']:.2f}")
    print(f"Min Steps: {results['min_steps']}")
    print(f"Max Steps: {results['max_steps']}")
    print(f"Training Time: {training_time:.2f} seconds")

    save_model(model)
    save_metrics(results, training_time)
    save_training_plot(callback)

    print("\nDone.")


if __name__ == "__main__":
    main()