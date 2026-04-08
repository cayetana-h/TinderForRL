import gymnasium as gym
import numpy as np
import yaml
import os
import sys
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "agents", "agent_qtable.py"))

spec = importlib.util.spec_from_file_location("agent_qtable", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
QTableAgent = agent_mod.QTableAgent
print("Agent loaded from:", AGENT_PATH)



def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def shape_reward(obs, obs_next, gamma, scale):
    """
    Potential-based reward shaping using position as potential.

    phi(s) = position of the car
    shaped_reward = r + gamma * phi(s') - phi(s)

    This gives a positive signal whenever the car moves right (toward goal),
    and a negative signal when it moves left — but crucially, the agent
    learns that building momentum (going left first) is worth the short
    term negative shaping because it gains velocity.

    The scale needs to be large enough to overcome the -1/step base reward.
    A scale of 100-150 works well empirically for this environment.
    """
    phi_current = -obs[0]       # current potential
    phi_next = -obs_next[0]     # next potential
    return phi_next - phi_current


def train():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "qtable_continuous.yaml"
    )
    config = load_config(config_path)

    env = gym.make("MountainCar-v0")

    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=config["num_bins"],
        num_actions=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
    )

    use_shaping = config.get("use_reward_shaping", True)
    shaping_scale = config.get("shaping_scale", 100.0)
    use_cost_penalty = config.get("use_cost_penalty", False)
    cost_coefficient = config.get("cost_coefficient", 0.1)

    rewards_per_episode = []
    action_counts_per_episode = []
    costs_per_episode = []
    successes = 0

    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        state = agent.discretize_state(obs)
        total_reward = 0
        action_count = 0
        total_cost = 0

        for step in range(config["max_steps"]):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if use_shaping:
                bonus = shape_reward(obs, obs_next, agent.gamma, shaping_scale)
                reward = reward + shaping_scale * bonus

            if use_cost_penalty and action != 1:  # Assuming action 1 is neutral
                reward -= cost_coefficient
                action_count += 1
                total_cost += cost_coefficient

            next_state = agent.discretize_state(obs_next)
            agent.update(state, action, reward, next_state, done)

            obs = obs_next
            state = next_state
            total_reward += reward

            if done:
                if terminated:
                    successes += 1
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        action_counts_per_episode.append(action_count)
        costs_per_episode.append(total_cost)

        if episode % 500 == 0:
            recent = rewards_per_episode[-100:] if episode >= 100 else rewards_per_episode
            print(
                f"Ep {episode:5d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg(last 100): {np.mean(recent):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Successes: {successes} | "
                f"Actions: {action_count} | "
                f"Cost: {total_cost:.2f}"
            )

    results_dir = os.path.join(CURRENT_DIR, "..", "results")
    models_dir = os.path.join(results_dir, "models")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    agent.save(os.path.join(models_dir, "continuous_q_table.npy"))
    np.save(os.path.join(metrics_dir, "continuous_rewards.npy"), np.array(rewards_per_episode))
    np.save(os.path.join(metrics_dir, "continuous_action_counts.npy"), np.array(action_counts_per_episode))
    np.save(os.path.join(metrics_dir, "continuous_costs.npy"), np.array(costs_per_episode))

    print(f"\nTraining done. Total successes: {successes}/{config['num_episodes']}")
    print(f"Q-table shape: {agent.q_table.shape} | Max Q: {np.max(agent.q_table):.4f}")

    # Test the learned policy
    print("\nTesting learned policy...")
    successes_test = 0
    epsilon_backup = agent.epsilon
    agent.epsilon = 0.0  # Force greedy
    for test_ep in range(10):
        obs, _ = env.reset()
        state = agent.discretize_state(obs)
        for step in range(config["max_steps"]):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = agent.discretize_state(obs_next)
            if done:
                if terminated:
                    successes_test += 1
                break
    agent.epsilon = epsilon_backup
    print(f"Test successes: {successes_test}/10")


if __name__ == "__main__":
    train()