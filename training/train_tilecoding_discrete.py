"""Training script for Tile Coding Q-Learning agent on MountainCar-v0."""

import gymnasium as gym
import numpy as np
import os
import sys
import importlib.util
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "agents", "agent_tilecoding.py"))

spec = importlib.util.spec_from_file_location("agent_tilecoding", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
TileCodingQAgent = agent_mod.TileCodingQAgent
print("Agent loaded from:", AGENT_PATH)


def shape_reward(obs, obs_next, gamma):
    """
    Velocity-based reward shaping.

    Incentivizes building speed (momentum) which is necessary to escape
    the valley in MountainCar.
    """
    return gamma * abs(obs_next[1]) - abs(obs[1])


def train():
    """Train tile coding agent with default configuration."""
    # Default configuration
    config = {
        "num_tilings": 8,
        "tiles_per_dim": [8, 8],
        "learning_rate": 0.01,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9995,
        "num_episodes": 10000,
        "max_steps": 500,
        "use_reward_shaping": True,
        "shaping_scale": 300.0,
    }

    # Initialize environment and agent
    env = gym.make("MountainCar-v0")

    agent = TileCodingQAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_actions=env.action_space.n,
        num_tilings=config["num_tilings"],
        tiles_per_dim=config["tiles_per_dim"],
        alpha=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
    )

    # Training setup
    use_shaping = config["use_reward_shaping"]
    shaping_scale = config["shaping_scale"]

    raw_rewards = []
    shaped_rewards = []
    step_counts = []
    successes = 0

    # TensorBoard logging
    writer = SummaryWriter(log_dir="runs/tilecoding_discrete")

    print("\nStarting Tile Coding Q-Learning Training...")
    print(f"Episodes: {config['num_episodes']}")
    print(f"Tilings: {config['num_tilings']}, Tiles per dim: {config['tiles_per_dim']}")
    print(f"Learning rate: {config['learning_rate']}, Gamma: {config['gamma']}")

    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        total_raw = 0
        total_shaped = 0
        episode_steps = 0

        for step in range(config["max_steps"]):
            action = agent.select_action(obs, training=True)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            raw_r = reward

            if use_shaping and not terminated:
                bonus = shape_reward(obs, obs_next, agent.gamma)
                reward = reward + shaping_scale * bonus

            # Update agent
            agent.update(obs, action, reward, obs_next, done)

            obs = obs_next
            total_raw += raw_r
            total_shaped += reward
            episode_steps += 1

            if done:
                if terminated:
                    successes += 1
                break

        agent.decay_epsilon()
        raw_rewards.append(total_raw)
        shaped_rewards.append(total_shaped)
        step_counts.append(episode_steps)

        # Log to TensorBoard
        writer.add_scalar("Reward/raw", total_raw, episode)
        writer.add_scalar("Reward/shaped", total_shaped, episode)
        writer.add_scalar("Metrics/steps", episode_steps, episode)
        writer.add_scalar("Metrics/epsilon", agent.epsilon, episode)

        if episode % 500 == 0:
            recent_raw = raw_rewards[-100:] if episode >= 100 else raw_rewards
            print(
                f"Ep {episode:5d} | "
                f"Raw avg: {np.mean(recent_raw):7.1f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Successes: {successes}"
            )

        # Greedy evaluation every 2000 episodes
        if episode % 2000 == 0 and episode > 0:
            eval_env = gym.make("MountainCar-v0")
            eval_wins = 0
            for _ in range(20):
                o, _ = eval_env.reset()
                for _ in range(500):
                    a = agent.select_action(o, training=False)
                    o, _, term, trunc, _ = eval_env.step(a)
                    if term:
                        eval_wins += 1
                        break
                    if trunc:
                        break
            print(f"  → Greedy eval: {eval_wins}/20")
            writer.add_scalar("Evaluation/greedy_wins", eval_wins, episode)
            eval_env.close()

    writer.close()

    # Save results
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    agent.save("results/models/tilecoding_weights.npy")
    np.save("results/metrics/tilecoding_rewards.npy", np.array(raw_rewards))
    np.save("results/metrics/tilecoding_rewards_shaped.npy", np.array(shaped_rewards))
    np.save("results/metrics/tilecoding_steps.npy", np.array(step_counts))

    print(f"\nTraining complete!")
    print(f"  Successes: {successes}/{config['num_episodes']}")
    print(f"  Weights saved to: results/models/tilecoding_weights.npy")
    print(f"  TensorBoard logs saved to: runs/tilecoding_discrete")


if __name__ == "__main__":
    train()
