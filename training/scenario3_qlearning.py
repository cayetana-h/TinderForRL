"""
Scenario 3: Discrete MountainCar, Minimum Fuel

Environment: MountainCar-v0
Objective: Reach the goal while minimizing fuel usage (non-neutral actions)
Algorithm: Q-learning with Q-table
Reward: Shaped reward with progress, velocity, fuel penalty, and goal bonus

Outputs saved:
- results/scenario3/models/scenario3_qtable.pkl
- results/scenario3/metrics/scenario3_qtable.json
- results/scenario3/plots/scenario3_qtable_training.png
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SCENARIO = 3
ALGORITHM = "Q-table"
ENV_NAME = "MountainCar-v0"
OBJECTIVE = "minimum_fuel"

TRAIN_EPISODES = 30000
EVAL_EPISODES = 100

POSITION_BINS = 30
VELOCITY_BINS = 30

LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.99

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9997

ACTION_COST = 0.001
GOAL_BONUS = 100

MAX_STEPS = 200

SEED = 42

RESULTS_DIR = Path("results") / "scenario3"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard_logs"


# ============================================================
# SETUP
# ============================================================

def create_dirs():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


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

        # Shaped reward
        reward = -1.0
        reward += 100.0 * progress
        reward += 10.0 * abs(vel)

        if action != self.neutral_action:
            reward -= self.action_cost

        if terminated:
            reward += self.goal_bonus

        self.prev_pos = pos

        return obs, reward, terminated, truncated, info


def discretize_state(obs, state_low, state_high, bins):
    """Convert continuous state to discrete bin indices"""
    ratios = (obs - state_low) / (state_high - state_low)
    indices = (ratios * np.array(bins)).astype(int)
    return tuple(np.clip(indices, 0, np.array(bins) - 1))


def choose_action(q_table, state, epsilon, env):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_table[state]))


# ============================================================
# TRAINING
# ============================================================

def train_qtable():
    np.random.seed(SEED)
    
    env = gym.make(ENV_NAME)
    env = DiscreteActionCostWrapper(env, action_cost=ACTION_COST, goal_bonus=GOAL_BONUS)
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    bins = [POSITION_BINS, VELOCITY_BINS]
    q_table = np.zeros((bins[0], bins[1], env.action_space.n))

    state_low = env.observation_space.low
    state_high = env.observation_space.high

    episode_rewards = []
    episode_steps = []
    episode_fuel = []
    success_history = []

    epsilon = EPSILON_START

    start_time = time.time()

    for episode in range(TRAIN_EPISODES):
        obs, _ = env.reset()
        state = discretize_state(obs, state_low, state_high, bins)

        total_reward = 0
        steps = 0
        fuel = 0
        success = False

        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon, env)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs, state_low, state_high, bins)
            done = terminated or truncated

            # Track fuel usage
            if action != 1:  # Non-neutral action
                fuel += 1

            # Q-learning update
            if terminated:
                target = reward
                success = True
            else:
                target = reward + DISCOUNT_FACTOR * np.max(q_table[next_state])

            q_table[state][action] += LEARNING_RATE * (target - q_table[state][action])

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_fuel.append(fuel)
        success_history.append(1 if success else 0)

        if (episode + 1) % 1000 == 0:
            recent_success = np.mean(success_history[-100:]) * 100
            recent_steps = np.mean(episode_steps[-100:])
            recent_reward = np.mean(episode_rewards[-100:])
            recent_fuel = np.mean(episode_fuel[-100:])

            print(
                f"Episode {episode + 1}/{TRAIN_EPISODES} | "
                f"Success: {recent_success:.1f}% | "
                f"Avg Steps: {recent_steps:.1f} | "
                f"Avg Fuel: {recent_fuel:.1f} | "
                f"Avg Reward: {recent_reward:.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    training_time = time.time() - start_time
    env.close()

    return {
        "q_table": q_table,
        "state_low": state_low,
        "state_high": state_high,
        "bins": bins,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_fuel": episode_fuel,
        "success_history": success_history,
        "training_time": training_time
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_qtable(q_table, state_low, state_high, bins):
    env = gym.make(ENV_NAME)
    env = DiscreteActionCostWrapper(env, action_cost=ACTION_COST, goal_bonus=GOAL_BONUS)
    env.action_space.seed(SEED)
    env.reset(seed=SEED)

    rewards = []
    steps_list = []
    successes = []
    fuel_costs = []

    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        state = discretize_state(obs, state_low, state_high, bins)

        total_reward = 0
        steps = 0
        success = False
        fuel_cost = 0

        for step in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs, state_low, state_high, bins)
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

def save_model(q_table, state_low, state_high, bins):
    model_data = {
        "q_table": q_table,
        "state_low": state_low,
        "state_high": state_high,
        "bins": bins
    }

    model_path = MODELS_DIR / "scenario3_qtable.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(model_data, file)

    print(f"Model saved to {model_path}")


def save_metrics(evaluation_results, training_time):
    metrics = {
        "scenario": SCENARIO,
        "algorithm": ALGORITHM,
        "environment": ENV_NAME,
        "objective": OBJECTIVE,
        "training_episodes": TRAIN_EPISODES,
        "evaluation_episodes": EVAL_EPISODES,
        "seed": SEED,
        "success_rate": evaluation_results["success_rate"],
        "success_rate_percent": evaluation_results["success_rate"] * 100,
        "average_reward": evaluation_results["average_reward"],
        "average_steps": evaluation_results["average_steps"],
        "average_fuel": evaluation_results["average_fuel"],
        "min_steps": evaluation_results["min_steps"],
        "max_steps": evaluation_results["max_steps"],
        "training_time_seconds": training_time
    }

    metrics_path = METRICS_DIR / "scenario3_qtable.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to {metrics_path}")


def save_training_plot(episode_rewards, episode_steps, episode_fuel, success_history):
    window = 100

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

    fuel_smooth = np.convolve(
        episode_fuel,
        np.ones(window) / window,
        mode="valid"
    )

    success_smooth = np.convolve(
        success_history,
        np.ones(window) / window,
        mode="valid"
    ) * 100

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(rewards_smooth)
    plt.title("Scenario 3 Q-table Training Performance")
    plt.ylabel("Average Reward")

    plt.subplot(4, 1, 2)
    plt.plot(steps_smooth)
    plt.ylabel("Average Steps")

    plt.subplot(4, 1, 3)
    plt.plot(fuel_smooth)
    plt.ylabel("Average Fuel Usage")

    plt.subplot(4, 1, 4)
    plt.plot(success_smooth)
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Episode")

    plt.tight_layout()

    plot_path = PLOTS_DIR / "scenario3_qtable_training.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Training plot saved to {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    create_dirs()

    print("=" * 60)
    print("SCENARIO 3: DISCRETE MOUNTAINCAR - MINIMUM FUEL")
    print("ALGORITHM: Q-TABLE")
    print("=" * 60)

    training_results = train_qtable()

    q_table = training_results["q_table"]
    state_low = training_results["state_low"]
    state_high = training_results["state_high"]
    bins = training_results["bins"]
    episode_rewards = training_results["episode_rewards"]
    episode_steps = training_results["episode_steps"]
    episode_fuel = training_results["episode_fuel"]
    success_history = training_results["success_history"]
    training_time = training_results["training_time"]

    print("\nEvaluating trained Q-table...")
    evaluation_results = evaluate_qtable(
        q_table,
        state_low,
        state_high,
        bins
    )

    print("\nEvaluation Results:")
    print(f"Success Rate: {evaluation_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {evaluation_results['average_reward']:.2f}")
    print(f"Average Steps: {evaluation_results['average_steps']:.2f}")
    print(f"Average Fuel: {evaluation_results['average_fuel']:.2f}")
    print(f"Min Steps: {evaluation_results['min_steps']}")
    print(f"Max Steps: {evaluation_results['max_steps']}")
    print(f"Training Time: {training_time:.2f} seconds")

    save_model(q_table, state_low, state_high, bins)
    save_metrics(evaluation_results, training_time)
    save_training_plot(episode_rewards, episode_steps, episode_fuel, success_history)

    print("\nDone.")


if __name__ == "__main__":
    main()