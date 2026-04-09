import gymnasium as gym
import numpy as np
import yaml
import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "agents", "agent_td3.py"))

spec = importlib.util.spec_from_file_location("agent_td3", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
TD3Agent = agent_mod.TD3Agent
print("Agent loaded from:", AGENT_PATH)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def shape_reward(obs, obs_next, gamma):
    """
    Velocity-based potential shaping — same logic as the discrete agent.

    phi(s) = |velocity|
    F(s, s') = gamma * phi(s') - phi(s)

    Rewards gaining speed regardless of direction, incentivising the
    rocking strategy. No local optimum where standing still pays off.
    """
    return gamma * abs(obs_next[1]) - abs(obs[1])


def train():
    config_path = os.path.join(CURRENT_DIR, "..", "config", "td3_continuous.yaml")
    config = load_config(config_path)

    env = gym.make("MountainCarContinuous-v0")

    state_dim = env.observation_space.shape[0]   # 2: position, velocity
    action_dim = env.action_space.shape[0]        # 1: continuous force in [-1, 1]
    max_action = float(env.action_space.high[0])  # 1.0

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        gamma=config["gamma"],
        tau=config["tau"],
        policy_noise=config["policy_noise"],
        noise_clip=config["noise_clip"],
        policy_delay=config["policy_delay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
    )

    use_shaping = config.get("use_reward_shaping", True)
    shaping_scale = config.get("shaping_scale", 50.0)
    exploration_noise = config["exploration_noise"]
    warmup_steps = config["warmup_steps"]

    rewards_per_episode = []
    costs_per_episode = []      # sum of 0.1 * action^2 — the intensity cost
    steps_per_episode = []
    successes = 0
    total_steps = 0

    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        total_reward = 0.0
        total_cost = 0.0
        episode_steps = 0

        for step in range(config["max_steps"]):
            total_steps += 1

            # Warmup: random actions before the replay buffer has enough data
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, noise_std=exploration_noise)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track the built-in action cost separately (env gives -0.1*a^2 per step)
            action_cost = 0.1 * float(action[0] ** 2)
            total_cost += action_cost

            # Optionally layer on velocity-based shaping
            train_reward = reward
            if use_shaping and not terminated:
                bonus = shape_reward(obs, obs_next, agent.gamma)
                train_reward = reward + shaping_scale * bonus

            agent.replay_buffer.add(obs, action, train_reward, obs_next, done)
            agent.train_step()

            obs = obs_next
            total_reward += reward   # always log raw env reward
            episode_steps += 1

            if done:
                if terminated:
                    successes += 1
                break

        rewards_per_episode.append(total_reward)
        costs_per_episode.append(total_cost)
        steps_per_episode.append(episode_steps)

        if episode % 50 == 0:
            recent = rewards_per_episode[-20:] if episode >= 20 else rewards_per_episode
            print(
                f"Ep {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg(last 20): {np.mean(recent):8.2f} | "
                f"Cost: {total_cost:.3f} | "
                f"Steps: {episode_steps} | "
                f"Successes: {successes} | "
                f"Buffer: {agent.replay_buffer.size}"
            )

    # Save everything
    results_dir = os.path.join(CURRENT_DIR, "..", "results")
    models_dir = os.path.join(results_dir, "models")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    agent.save(os.path.join(models_dir, "td3_continuous.pt"))
    np.save(os.path.join(metrics_dir, "td3_rewards.npy"), np.array(rewards_per_episode))
    np.save(os.path.join(metrics_dir, "td3_costs.npy"), np.array(costs_per_episode))
    np.save(os.path.join(metrics_dir, "td3_steps.npy"), np.array(steps_per_episode))

    print(f"\nTraining done. Total successes: {successes}/{config['num_episodes']}")

    # Quick greedy test
    print("\nTesting greedy policy (10 episodes)...")
    test_successes = 0
    for _ in range(10):
        obs, _ = env.reset()
        for _ in range(config["max_steps"]):
            action = agent.select_action(obs, noise_std=0.0)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                test_successes += 1
                break
            if truncated:
                break
    print(f"Test successes: {test_successes}/10")


if __name__ == "__main__":
    train()
