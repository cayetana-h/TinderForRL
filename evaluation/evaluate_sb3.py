from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3

from utils.io import ensure_dir, load_yaml, save_csv_rows, save_json
from utils.metrics import summarize_episodes
from utils.wrappers import ContinuousExtraActionCostWrapper


ROOT = Path(__file__).resolve().parents[1]


def make_env(seed: int, extra_cost: float = 0.0):
    env = gym.make("MountainCarContinuous-v0")
    env = ContinuousExtraActionCostWrapper(env, extra_cost_coefficient=extra_cost)
    env.reset(seed=seed)
    return env


def evaluate(model_path: str | Path, algo: str, config_path: str | Path, experiment_name: str, episodes: int | None = None):
    cfg = load_yaml(config_path)
    eval_episodes = int(episodes if episodes is not None else cfg.get("eval_episodes", 20))
    extra_cost = float(cfg.get("extra_action_cost", 0.0))
    max_steps = 1000  # MountainCarContinuous default horizon

    if algo.lower() == "td3":
        model = TD3.load(model_path)
    elif algo.lower() == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError("algo must be 'td3' or 'sac'")

    env = make_env(seed=int(cfg.get("seed", 42)), extra_cost=extra_cost)

    rewards, steps_list, costs, successes = [], [], [], []
    episode_rows = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        total_reward = 0.0
        total_cost = 0.0
        success = False
        step_count = 0

        for step_count in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_cost += float(info.get("extra_action_cost", 0.0))

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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", type=str, choices=["td3", "sac"], required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deeprl.yaml"))
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.model, args.algo, args.config, args.name, args.episodes)


if __name__ == "__main__":
    main()