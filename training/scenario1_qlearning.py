"""
Scenario 1: Discrete MountainCar, Minimum Steps


Environment: MountainCar-v0
Objective: Reach the goal as quickly as possible
Algorithm: Q-learning with Q-table
Reward: Shaped reward with goal bonus (+100)


Outputs saved:
- results/models/scenario1_qtable.pkl
- results/metrics/scenario1_qtable.json
- results/plots/scenario1_qtable_training.png
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


SCENARIO = 1
ALGORITHM = "Q-table"
ENV_NAME = "MountainCar-v0"
OBJECTIVE = "minimum_steps"


TRAIN_EPISODES = 25000
EVAL_EPISODES = 100


POSITION_BINS = 40
VELOCITY_BINS = 40


LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99


EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9997


MAX_STEPS = 200


SEED = 42  # Added for reproducibility


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




def discretize_state(state, position_bins, velocity_bins):
    position = state[0]
    velocity = state[1]


    position_index = np.digitize(position, position_bins) - 1
    velocity_index = np.digitize(velocity, velocity_bins) - 1


    position_index = np.clip(position_index, 0, POSITION_BINS - 1)
    velocity_index = np.clip(velocity_index, 0, VELOCITY_BINS - 1)


    return position_index, velocity_index




def choose_action(q_table, discrete_state, epsilon, env):
    if np.random.random() < epsilon:
        return env.action_space.sample()


    return int(np.argmax(q_table[discrete_state]))




# ============================================================
# TRAINING
# ============================================================


def train_qtable():
    # Set seeds for reproducibility
    np.random.seed(SEED)
    
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)


    position_bins = np.linspace(
        env.observation_space.low[0],
        env.observation_space.high[0],
        POSITION_BINS
    )


    velocity_bins = np.linspace(
        env.observation_space.low[1],
        env.observation_space.high[1],
        VELOCITY_BINS
    )


    q_table = np.zeros((POSITION_BINS, VELOCITY_BINS, env.action_space.n))


    episode_rewards = []
    episode_steps = []
    success_history = []


    epsilon = EPSILON_START


    start_time = time.time()


    for episode in range(TRAIN_EPISODES):
        state, _ = env.reset()
        discrete_state = discretize_state(state, position_bins, velocity_bins)


        total_reward = 0
        steps = 0
        success = False


        for step in range(MAX_STEPS):
            action = choose_action(q_table, discrete_state, epsilon, env)


            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


            next_discrete_state = discretize_state(
                next_state,
                position_bins,
                velocity_bins
            )


            # Give a strong bonus for reaching the goal.
            # This helps Q-learning learn faster because default MountainCar rewards are sparse.
            shaped_reward = reward
            if terminated:
                shaped_reward = 100


            old_q_value = q_table[discrete_state][action]
            next_max_q_value = np.max(q_table[next_discrete_state])


            new_q_value = old_q_value + LEARNING_RATE * (
                shaped_reward + DISCOUNT_FACTOR * next_max_q_value - old_q_value
            )


            q_table[discrete_state][action] = new_q_value


            discrete_state = next_discrete_state
            total_reward += reward
            steps += 1


            if terminated:
                success = True
                break


            if done:
                break


        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        success_history.append(1 if success else 0)


        if (episode + 1) % 1000 == 0:
            recent_success = np.mean(success_history[-100:]) * 100
            recent_steps = np.mean(episode_steps[-100:])
            recent_reward = np.mean(episode_rewards[-100:])


            print(
                f"Episode {episode + 1}/{TRAIN_EPISODES} | "
                f"Success: {recent_success:.1f}% | "
                f"Avg Steps: {recent_steps:.1f} | "
                f"Avg Reward: {recent_reward:.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )


    training_time = time.time() - start_time
    env.close()


    return {
        "q_table": q_table,
        "position_bins": position_bins,
        "velocity_bins": velocity_bins,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "success_history": success_history,
        "training_time": training_time
    }




# ============================================================
# EVALUATION
# ============================================================


def evaluate_qtable(q_table, position_bins, velocity_bins):
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)  # Seed for consistent evaluation


    rewards = []
    steps_list = []
    successes = []
    fuel_costs = []


    for episode in range(EVAL_EPISODES):
        state, _ = env.reset()
        discrete_state = discretize_state(state, position_bins, velocity_bins)


        total_reward = 0
        steps = 0
        success = False
        fuel_cost = 0


        for step in range(MAX_STEPS):
            action = int(np.argmax(q_table[discrete_state]))


            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


            discrete_state = discretize_state(
                next_state,
                position_bins,
                velocity_bins
            )


            total_reward += reward
            steps += 1


            # For discrete MountainCar, pushing left/right uses fuel.
            # Action 1 means no push.
            if action != 1:
                fuel_cost += 1


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


def save_model(q_table, position_bins, velocity_bins):
    model_data = {
        "q_table": q_table,
        "position_bins": position_bins,
        "velocity_bins": velocity_bins
    }


    model_path = MODELS_DIR / "scenario1_qtable.pkl"


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
        "seed": SEED,  # Added
        "success_rate": evaluation_results["success_rate"],
        "success_rate_percent": evaluation_results["success_rate"] * 100,
        "average_reward": evaluation_results["average_reward"],
        "average_steps": evaluation_results["average_steps"],
        "average_fuel": evaluation_results["average_fuel"],
        "min_steps": evaluation_results["min_steps"],
        "max_steps": evaluation_results["max_steps"],
        "training_time_seconds": training_time
    }


    metrics_path = METRICS_DIR / "scenario1_qtable.json"


    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)


    print(f"Metrics saved to {metrics_path}")




def save_training_plot(episode_rewards, episode_steps, success_history):
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


    success_smooth = np.convolve(
        success_history,
        np.ones(window) / window,
        mode="valid"
    ) * 100


    plt.figure(figsize=(12, 8))


    plt.subplot(3, 1, 1)
    plt.plot(rewards_smooth)
    plt.title("Scenario 1 Q-table Training Performance")
    plt.ylabel("Average Reward")


    plt.subplot(3, 1, 2)
    plt.plot(steps_smooth)
    plt.ylabel("Average Steps")


    plt.subplot(3, 1, 3)
    plt.plot(success_smooth)
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Episode")


    plt.tight_layout()


    plot_path = PLOTS_DIR / "scenario1_qtable_training.png"
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
    print("ALGORITHM: Q-TABLE")
    print("=" * 60)


    training_results = train_qtable()


    q_table = training_results["q_table"]
    position_bins = training_results["position_bins"]
    velocity_bins = training_results["velocity_bins"]
    episode_rewards = training_results["episode_rewards"]
    episode_steps = training_results["episode_steps"]
    success_history = training_results["success_history"]
    training_time = training_results["training_time"]


    print("\nEvaluating trained Q-table...")
    evaluation_results = evaluate_qtable(
        q_table,
        position_bins,
        velocity_bins
    )


    print("\nEvaluation Results:")
    print(f"Success Rate: {evaluation_results['success_rate'] * 100:.2f}%")
    print(f"Average Reward: {evaluation_results['average_reward']:.2f}")
    print(f"Average Steps: {evaluation_results['average_steps']:.2f}")
    print(f"Average Fuel: {evaluation_results['average_fuel']:.2f}")
    print(f"Min Steps: {evaluation_results['min_steps']}")
    print(f"Max Steps: {evaluation_results['max_steps']}")
    print(f"Training Time: {training_time:.2f} seconds")


    save_model(q_table, position_bins, velocity_bins)
    save_metrics(evaluation_results, training_time)
    save_training_plot(episode_rewards, episode_steps, success_history)


    print("\nDone.")




if __name__ == "__main__":
    main()