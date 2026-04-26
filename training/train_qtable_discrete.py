from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# Add project root to sys.path for module imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.agent_qtable import QTableAgent
from utils.io import ensure_dir, load_yaml, save_csv_rows, save_json
from utils.metrics import rolling_mean, summarize_episodes

ROOT = Path(__file__).resolve().parents[1]


def shape_reward(obs, obs_next, gamma, scale):
    """
    Potential-based shaping using velocity magnitude.
    """
    phi_current = abs(float(obs[1]))
    phi_next = abs(float(obs_next[1]))
    return scale * (gamma * phi_next - phi_current)


def train(config_path: str | Path = ROOT / "config" / "qtable_discrete.yaml"):
    print("[DEBUG] Loading config...")
    config = load_yaml(config_path)
    print(f"[DEBUG] Config loaded: {config}")
    seed = int(config.get("seed", 42))
    np.random.seed(seed)


    env = gym.make("MountainCar-v0")
    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=config["num_bins"],
        num_actions=env.action_space.n,
    )

    # Restore standard Q-learning training loop
    num_episodes = config.get("num_episodes", 10000)
    max_steps = config.get("max_steps", 200)
    gamma = config.get("gamma", 0.99)
    epsilon_start = config.get("epsilon_start", 1.0)
    epsilon_end = config.get("epsilon_end", 0.01)
    epsilon_decay = config.get("epsilon_decay", 0.9995)
    use_shaping = config.get("use_reward_shaping", True)
    shaping_scale = config.get("shaping_scale", 300.0)
    eval_episodes = config.get("eval_episodes", 20)

    # Ensure output directories exist
    model_dir = ensure_dir(ROOT / "results" / "models")
    print(f"[DEBUG] Model dir: {model_dir}")
    metrics_dir = ensure_dir(ROOT / "results" / "metrics" / "qtable_discrete")
    print(f"[DEBUG] Metrics dir: {metrics_dir}")
    comparison_dir = ensure_dir(ROOT / "results" / "comparison")
    print(f"[DEBUG] Comparison dir: {comparison_dir}")

    writer = SummaryWriter("runs/qtable_discrete")

    rewards = []
    shaped_rewards = []
    steps_list = []
    costs = []
    successes = []

    print(f"[DEBUG] Starting Q-table training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        if episode == 0:
            print("[DEBUG] Entered training loop. First episode starting...")
        obs, _ = env.reset()
        state = agent.discretize_state(obs)
        total_reward = 0.0
        total_shaped = 0.0
        total_cost = 0.0
        step_count = 0
        terminated = False

        for step in range(max_steps):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_reward = reward
            shaped_reward = reward
            if use_shaping and not terminated:
                bonus = shape_reward(obs, obs_next, gamma, shaping_scale)
                shaped_reward = reward + bonus

            agent.update(state, action, shaped_reward, agent.discretize_state(obs_next), done)

            obs = obs_next
            state = agent.discretize_state(obs)
            total_reward += raw_reward
            total_shaped += shaped_reward
            step_count += 1

            if done:
                break

        rewards.append(total_reward)
        shaped_rewards.append(total_shaped)
        steps_list.append(step_count)
        costs.append(total_cost)
        successes.append(terminated)

        agent.decay_epsilon()

        # Log to TensorBoard
        writer.add_scalar("Reward/raw", total_reward, episode)
        writer.add_scalar("Reward/shaped", total_shaped, episode)
        writer.add_scalar("Metrics/steps", step_count, episode)
        writer.add_scalar("Metrics/epsilon", agent.epsilon, episode)

        if episode % 500 == 0:
            recent = rewards[-100:] if len(rewards) >= 100 else rewards
            print(
                f"Ep {episode:5d} | "
                f"raw avg(last 100): {np.mean(recent):8.2f} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"successes: {sum(successes)}"
            )

    writer.close()

    agent.save(model_dir / "q_table.npy")

    rows = []
    for i, (raw_r, shaped_r, steps, cost, succ) in enumerate(
        zip(rewards, shaped_rewards, steps_list, costs, successes),
        start=1,
    ):
        rows.append(
            {
                "episode": i,
                "raw_reward": raw_r,
                "shaped_reward": shaped_r,
                "steps": steps,
                "cost": cost,
                "success": int(succ),
                "epsilon": float(agent.epsilon),
            }
        )

    save_csv_rows(rows, metrics_dir / "episode_metrics.csv")
    save_json(summarize_episodes(rewards, steps_list, costs, successes), metrics_dir / "summary.json")
    np.save(metrics_dir / "rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(metrics_dir / "shaped_rewards.npy", np.asarray(shaped_rewards, dtype=np.float32))
    np.save(metrics_dir / "steps.npy", np.asarray(steps_list, dtype=np.int32))
    np.save(metrics_dir / "costs.npy", np.asarray(costs, dtype=np.float32))

    rm = rolling_mean(rewards, 100)
    if rm.size:
        np.save(comparison_dir / "qtable_discrete_rolling_mean.npy", rm)

    print(f"Training done. Successes: {sum(successes)}/{num_episodes}")
    print(f"Q-table shape: {agent.q_table.shape} | Max Q: {np.max(agent.q_table):.4f}")
    print(f"Metrics and model saved to results/metrics/qtable_discrete/ and results/models/")

    # Greedy evaluation
    greedy_agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=config["num_bins"],
        num_actions=env.action_space.n,
    )
    greedy_agent.q_table = agent.q_table.copy()

    eval_success = 0
    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        state = greedy_agent.discretize_state(obs)

        for _ in range(max_steps):
            action = greedy_agent.greedy_action(state)
            obs, _, terminated, truncated, _ = env.step(action)
            state = greedy_agent.discretize_state(obs)

            if terminated:
                eval_success += 1
                break
            if truncated:
                break

    print(f"Greedy eval: {eval_success}/{eval_episodes}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "qtable_discrete.yaml"))
    args = parser.parse_args()
    train(args.config)