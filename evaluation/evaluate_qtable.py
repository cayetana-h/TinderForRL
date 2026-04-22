from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from agents.agent_qtable import QTableAgent
from utils.io import ensure_dir, load_yaml, save_csv_rows, save_json
from utils.metrics import summarize_episodes


ROOT = Path(__file__).resolve().parents[1]


def evaluate(model_path: str | Path, config_path: str | Path, experiment_name: str, episodes: int | None = None):
    cfg = load_yaml(config_path)
    max_steps = int(cfg.get("max_steps", 200))
    eval_episodes = int(episodes if episodes is not None else cfg.get("eval_episodes", 20))

    env = gym.make("MountainCar-v0")
    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=cfg["num_bins"],
        num_actions=env.action_space.n,
    )
    agent.load(model_path)
    agent.epsilon = 0.0

    rewards, steps_list, costs, successes = [], [], [], []
    episode_rows = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        state = agent.discretize_state(obs)

        total_reward = 0.0
        total_cost = 0.0
        step_count = 0
        success = False

        for step_count in range(1, max_steps + 1):
            action = agent.greedy_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = agent.discretize_state(obs)

            total_reward += float(reward)
            if terminated:
                success = True
                break
            if truncated:
                break

        rewards.append(total_reward)
        steps_list.append(step_count)
        costs.append(total_cost)
        successes.append(success)
        episode_rows.append(
            {
                "episode": ep + 1,
                "reward": total_reward,
                "steps": step_count,
                "cost": total_cost,
                "success": int(success),
            }
        )

    metrics_dir = ensure_dir(ROOT / "results" / "metrics" / experiment_name)
    save_csv_rows(episode_rows, metrics_dir / "evaluation.csv")
    summary = summarize_episodes(rewards, steps_list, costs, successes)
    save_json(summary, metrics_dir / "summary.json")

    print(f"Saved evaluation to {metrics_dir}")
    print(summary)
    env.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(ROOT / "results" / "models" / "qtable_discrete.npy"))
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "qtable_discrete.yaml"))
    parser.add_argument("--name", type=str, default="qtable_discrete")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.model, args.config, args.name, args.episodes)


if __name__ == "__main__":
    main()