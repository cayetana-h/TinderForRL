from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from agents.agent_qtable import QTableAgent
from utils.io import ensure_dir, load_yaml, save_csv_rows, save_json
from utils.metrics import summarize_episodes
from utils.wrappers import DiscreteActionCostWrapper


ROOT = Path(__file__).resolve().parents[1]


def shape_reward(obs, obs_next, gamma, scale):
    # Still potential-based; useful if you want to test shaping against action-cost reward.
    phi_current = float(obs[0])
    phi_next = float(obs_next[0])
    return scale * (gamma * phi_next - phi_current)


def train(config_path: str | Path = ROOT / "config" / "qtable_continuous.yaml"):
    config = load_yaml(config_path)
    np.random.seed(int(config.get("seed", 42)))

    base_env = gym.make("MountainCar-v0")
    env = DiscreteActionCostWrapper(base_env, cost_coefficient=float(config.get("cost_coefficient", 0.1)))

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

    use_shaping = bool(config.get("use_reward_shaping", False))
    shaping_scale = float(config.get("shaping_scale", 1000.0))
    num_episodes = int(config["num_episodes"])
    max_steps = int(config["max_steps"])
    eval_episodes = int(config.get("eval_episodes", 20))

    rewards = []
    shaped_rewards = []
    steps_list = []
    costs = []
    successes = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=int(config.get("seed", 42)) + episode)
        state = agent.discretize_state(obs)

        total_reward = 0.0
        total_shaped = 0.0
        total_cost = 0.0
        done = False
        step_count = 0

        for step_count in range(1, max_steps + 1):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            raw_reward = float(reward)
            shaped_reward = raw_reward
            extra_cost = float(info.get("extra_action_cost", 0.0))
            total_cost += extra_cost

            if use_shaping and not terminated:
                shaped_reward = raw_reward + shape_reward(obs, obs_next, agent.gamma, shaping_scale)

            agent.update(
                state=state,
                action=action,
                reward=shaped_reward,
                next_state=agent.discretize_state(obs_next),
                done=done,
            )

            obs = obs_next
            state = agent.discretize_state(obs_next)
            total_reward += raw_reward
            total_shaped += shaped_reward

            if done:
                break

        agent.decay_epsilon()

        rewards.append(total_reward)
        shaped_rewards.append(total_shaped)
        steps_list.append(step_count)
        costs.append(total_cost)
        successes.append(bool(done and terminated))

        if episode % 500 == 0:
            recent = rewards[-100:] if len(rewards) >= 100 else rewards
            print(
                f"Ep {episode:5d} | "
                f"raw avg(last 100): {np.mean(recent):8.2f} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"successes: {sum(successes)} | "
                f"cost: {total_cost:.2f}"
            )

    model_dir = ensure_dir(ROOT / "results" / "models")
    metrics_dir = ensure_dir(ROOT / "results" / "metrics" / "qtable_action_cost")

    agent.save(model_dir / "qtable_action_cost.npy")

    rows = []
    for i, (r, sr, s, c, succ) in enumerate(zip(rewards, shaped_rewards, steps_list, costs, successes), start=1):
        rows.append(
            {
                "episode": i,
                "raw_reward": r,
                "shaped_reward": sr,
                "steps": s,
                "cost": c,
                "success": int(succ),
                "epsilon": float(agent.epsilon),
            }
        )

    save_csv_rows(rows, metrics_dir / "episode_metrics.csv")
    save_json(summarize_episodes(rewards, steps_list, costs, successes), metrics_dir / "summary.json")
    np.save(metrics_dir / "rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(metrics_dir / "shaped_rewards.npy", np.asarray(shaped_rewards, dtype=np.float32))
    np.save(metrics_dir / "costs.npy", np.asarray(costs, dtype=np.float32))

    print(f"Training done. Successes: {sum(successes)}/{num_episodes}")
    print(f"Q-table shape: {agent.q_table.shape} | Max Q: {np.max(agent.q_table):.4f}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "qtable_continuous.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()