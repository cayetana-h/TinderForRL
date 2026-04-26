"""
Multi-seed evaluation framework for RL agents.

This module provides utilities for running agents across multiple random seeds,
computing statistics, and generating confidence intervals.
"""

import numpy as np
import gymnasium as gym
from scipy import stats
import os


class MultiSeedEvaluator:
    """Evaluator that runs an agent across multiple seeds with statistics."""

    def __init__(self, env_name, num_seeds=10, episodes_per_seed=100, max_steps=500):
        """
        Initialize evaluator.

        Args:
            env_name: Gymnasium environment name (e.g., 'MountainCar-v0')
            num_seeds: Number of random seeds to run
            episodes_per_seed: Episodes to evaluate per seed
            max_steps: Max steps per episode
        """
        self.env_name = env_name
        self.num_seeds = num_seeds
        self.episodes_per_seed = episodes_per_seed
        self.max_steps = max_steps

    def evaluate(self, agent_loader_fn, agent_path=None, verbose=True):
        """
        Run evaluation across multiple seeds.

        Args:
            agent_loader_fn: Function that takes (agent_path, seed) and returns loaded agent
            agent_path: Path to pre-trained agent (optional)
            verbose: Print progress

        Returns:
            Dict with aggregated statistics
        """
        all_rewards = []
        all_steps = []
        all_successes = []

        for seed in range(self.num_seeds):
            if verbose:
                print(f"Seed {seed + 1}/{self.num_seeds}... ", end="", flush=True)

            # Create environment with fixed seed
            env = gym.make(self.env_name)
            env.reset(seed=seed)
            np.random.seed(seed)

            # Load/create agent
            agent = agent_loader_fn(agent_path, seed)

            # Run episodes
            episode_rewards = []
            episode_steps = []
            successes = 0

            for ep in range(self.episodes_per_seed):
                obs, _ = env.reset()
                total_reward = 0
                steps = 0

                for t in range(self.max_steps):
                    # Greedy action (no exploration during evaluation)
                    action = agent.select_action(obs, training=False)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1

                    if terminated or truncated:
                        if terminated:
                            successes += 1
                        break

                episode_rewards.append(total_reward)
                episode_steps.append(steps)

            # Aggregate across episodes for this seed
            mean_reward = np.mean(episode_rewards)
            mean_steps = np.mean(episode_steps)
            success_rate = successes / self.episodes_per_seed

            all_rewards.append(mean_reward)
            all_steps.append(mean_steps)
            all_successes.append(success_rate)

            if verbose:
                print(f"Reward={mean_reward:.2f}, Steps={mean_steps:.1f}, Success={success_rate:.2f}")

            env.close()

        return self._compute_statistics(all_rewards, all_steps, all_successes)

    def _compute_statistics(self, rewards, steps, successes):
        """Compute mean, std, and 95% CI for metrics."""
        rewards = np.array(rewards)
        steps = np.array(steps)
        successes = np.array(successes)

        def ci_95(data):
            """Compute 95% confidence interval using t-distribution."""
            if len(data) < 2:
                return (data[0], data[0])
            return stats.t.interval(
                0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data)
            )

        return {
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "ci_lower": float(ci_95(rewards)[0]),
                "ci_upper": float(ci_95(rewards)[1]),
                "all": rewards.tolist(),
            },
            "steps": {
                "mean": float(np.mean(steps)),
                "std": float(np.std(steps)),
                "ci_lower": float(ci_95(steps)[0]),
                "ci_upper": float(ci_95(steps)[1]),
                "all": steps.tolist(),
            },
            "success_rate": {
                "mean": float(np.mean(successes)),
                "std": float(np.std(successes)),
                "all": successes.tolist(),
            },
        }

    @staticmethod
    def print_results(results, agent_name="Agent"):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULTS: {agent_name}")
        print("=" * 70)

        print(
            f"\nEpisode Reward (per seed average):"
            f"\n  Mean:     {results['reward']['mean']:8.2f}"
            f"\n  Std:      {results['reward']['std']:8.2f}"
            f"\n  95% CI:   [{results['reward']['ci_lower']:7.2f}, {results['reward']['ci_upper']:7.2f}]"
        )

        print(
            f"\nSteps to Goal (per seed average):"
            f"\n  Mean:     {results['steps']['mean']:8.2f}"
            f"\n  Std:      {results['steps']['std']:8.2f}"
            f"\n  95% CI:   [{results['steps']['ci_lower']:7.2f}, {results['steps']['ci_upper']:7.2f}]"
        )

        print(
            f"\nSuccess Rate (seeds):"
            f"\n  Mean:     {results['success_rate']['mean']:8.2%}"
            f"\n  Std:      {results['success_rate']['std']:8.2%}"
        )

        print("\n" + "=" * 70)

    @staticmethod
    def compare_agents(results_dict, metric="reward"):
        """
        Compare multiple agents and print ranked comparison.

        Args:
            results_dict: Dict mapping agent_name -> results
            metric: 'reward' or 'steps'
        """
        print("\n" + "=" * 70)
        print(f"COMPARISON: {metric.upper()}")
        print("=" * 70)

        agents = sorted(
            results_dict.items(),
            key=lambda x: x[1][metric]["mean"],
            reverse=(metric == "reward"),
        )

        for rank, (name, results) in enumerate(agents, 1):
            mean = results[metric]["mean"]
            std = results[metric]["std"]
            ci = results[metric]
            print(
                f"{rank}. {name:20s}: "
                f"{mean:8.2f} ± {std:6.2f} "
                f"[95% CI: {ci['ci_lower']:7.2f}, {ci['ci_upper']:7.2f}]"
            )

        print("=" * 70)


def evaluate_qtable_agent(q_table_path, num_bins, env_name="MountainCar-v0", num_seeds=10):
    """
    Convenience function to evaluate a trained Q-table agent.

    Args:
        q_table_path: Path to saved Q-table
        num_bins: [bins_pos, bins_vel] matching training
        env_name: Gymnasium environment
        num_seeds: Number of evaluation seeds

    Returns:
        Evaluation results dictionary
    """
    from agents.agent_qtable import QTableAgent

    def load_qtable(path, seed):
        env = gym.make(env_name)
        agent = QTableAgent(
            state_low=env.observation_space.low,
            state_high=env.observation_space.high,
            num_bins=num_bins,
            num_actions=env.action_space.n,
        )
        agent.load(path)
        agent.epsilon = 0.0  # Greedy during evaluation
        return agent

    evaluator = MultiSeedEvaluator(env_name, num_seeds=num_seeds)
    results = evaluator.evaluate(load_qtable, agent_path=q_table_path)

    MultiSeedEvaluator.print_results(results, agent_name="Q-Table Agent")

    return results


def evaluate_tilecoding_agent(
    weights_path, num_tilings=8, tiles_per_dim=None, env_name="MountainCar-v0", num_seeds=10
):
    """
    Convenience function to evaluate a trained tile coding agent.

    Args:
        weights_path: Path to saved tile coding weights
        num_tilings: Number of tilings
        tiles_per_dim: Tiles per dimension
        env_name: Gymnasium environment
        num_seeds: Number of evaluation seeds

    Returns:
        Evaluation results dictionary
    """
    from agents.agent_tilecoding import TileCodingQAgent

    def load_tilecoding(path, seed):
        env = gym.make(env_name)
        agent = TileCodingQAgent(
            state_low=env.observation_space.low,
            state_high=env.observation_space.high,
            num_actions=env.action_space.n,
            num_tilings=num_tilings,
            tiles_per_dim=tiles_per_dim,
        )
        agent.load(path)
        agent.epsilon = 0.0  # Greedy during evaluation
        return agent

    evaluator = MultiSeedEvaluator(env_name, num_seeds=num_seeds)
    results = evaluator.evaluate(load_tilecoding, agent_path=weights_path)

    MultiSeedEvaluator.print_results(results, agent_name="Tile Coding Agent")

    return results


if __name__ == "__main__":
    # Example: Compare Q-table and tile coding agents
    print("Multi-Seed Evaluation Example")

    results_q = evaluate_qtable_agent(
        q_table_path="results/models/q_table.npy",
        num_bins=[200, 200],
        num_seeds=10,
    )

    results_tc = evaluate_tilecoding_agent(
        weights_path="results/models/tilecoding_weights.npy",
        num_tilings=8,
        tiles_per_dim=[8, 8],
        num_seeds=10,
    )

    MultiSeedEvaluator.compare_agents(
        {"Q-Table": results_q, "Tile Coding": results_tc}, metric="reward"
    )
