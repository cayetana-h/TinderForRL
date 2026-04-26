"""
Example usage of new features: tile coding, energy augmentation, interpretability, and multi-seed eval.
"""

import os
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import new components
from agents.agent_tilecoding import TileCodingQAgent
from agents.agent_qtable import QTableAgent
from utils.wrappers import EnergyAugmentWrapper, CombinedAugmentationWrapper
from analysis.qtable_interpretability import analyze_qtable_policy
from evaluation.multi_seed_eval import (
    MultiSeedEvaluator,
    evaluate_qtable_agent,
    evaluate_tilecoding_agent,
)


def example_1_tile_coding_training():
    """Example: Train tile coding agent and log to TensorBoard."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Tile Coding Training with TensorBoard")
    print("=" * 70)

    env = gym.make("MountainCar-v0")

    agent = TileCodingQAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_actions=env.action_space.n,
        num_tilings=8,
        tiles_per_dim=[8, 8],
        alpha=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
    )

    writer = SummaryWriter(log_dir="runs/example_tilecoding")

    num_episodes = 500
    successes = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(500):
            action = agent.select_action(obs, training=True)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(obs, action, reward, obs_next, done)
            obs = obs_next
            total_reward += reward
            steps += 1

            if done:
                if terminated:
                    successes += 1
                break

        agent.decay_epsilon()

        # Log metrics
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Steps", steps, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={total_reward:.1f}, Steps={steps}, Success={successes}")

    writer.close()
    print(f"\nTraining complete! TensorBoard logs: runs/example_tilecoding")


def example_2_energy_augmented_observation():
    """Example: Use energy augmented wrapper."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Energy Augmented Observations")
    print("=" * 70)

    # Create environment with energy augmentation
    env = gym.make("MountainCar-v0")
    env = EnergyAugmentWrapper(env)

    obs, _ = env.reset()
    print(f"Original observation space: 2D (position, velocity)")
    print(f"Augmented observation space: {env.observation_space.shape}D")
    print(f"Sample augmented observation: {obs}")
    print(f"  position: {obs[0]:.4f}")
    print(f"  velocity: {obs[1]:.4f}")
    print(f"  kinetic energy: {obs[2]:.4f}")
    print(f"  potential energy: {obs[3]:.4f}")


def example_3_policy_interpretability():
    """Example: Extract and visualize policy using decision tree."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Policy Interpretability")
    print("=" * 70)

    # Check if Q-table exists
    q_table_path = "results/models/q_table.npy"
    if not os.path.exists(q_table_path):
        print(f"Q-table not found at {q_table_path}")
        print("Train Q-table first using: python training/train_qtable_discrete.py")
        return

    env = gym.make("MountainCar-v0")
    num_bins = [200, 200]
    env_bounds = {"low": env.observation_space.low, "high": env.observation_space.high}

    clf, states, actions = analyze_qtable_policy(
        q_table_path,
        num_bins,
        env_bounds,
        output_dir="results/interpretability",
        max_tree_depth=4,
    )

    print("\nPolicy extracted successfully!")
    print("Output files:")
    print("  - results/interpretability/policy_tree.png")
    print("  - results/interpretability/feature_importances.png")


def example_4_multiseed_evaluation():
    """Example: Run multi-seed evaluation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Multi-Seed Evaluation")
    print("=" * 70)

    # Check if models exist
    q_table_path = "results/models/q_table.npy"
    tilecoding_path = "results/models/tilecoding_weights.npy"

    models_available = []
    if os.path.exists(q_table_path):
        models_available.append(("Q-Table", q_table_path, "qtable"))
    if os.path.exists(tilecoding_path):
        models_available.append(("Tile Coding", tilecoding_path, "tilecoding"))

    if not models_available:
        print("No trained models found!")
        print("Train models first:")
        print("  python training/train_qtable_discrete.py")
        print("  python training/train_tilecoding_discrete.py")
        return

    results_dict = {}

    for name, path, model_type in models_available:
        print(f"\nEvaluating {name}...")
        if model_type == "qtable":
            results = evaluate_qtable_agent(
                q_table_path=path,
                num_bins=[200, 200],
                num_seeds=10,
            )
        else:
            results = evaluate_tilecoding_agent(
                weights_path=path,
                num_tilings=8,
                tiles_per_dim=[8, 8],
                num_seeds=10,
            )
        results_dict[name] = results

    # Compare agents
    if len(results_dict) > 1:
        MultiSeedEvaluator.compare_agents(results_dict, metric="reward")
        MultiSeedEvaluator.compare_agents(results_dict, metric="steps")


def example_5_combined_augmentation():
    """Example: Use combined augmentation wrapper."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Combined Augmentation Wrapper")
    print("=" * 70)

    env = gym.make("MountainCar-v0")
    env = CombinedAugmentationWrapper(env, add_energy=True)

    print(f"Original observation space: 2D")
    print(f"Augmented observation space: {env.observation_space.shape}D")

    obs, _ = env.reset()
    print(f"\nSample observation (augmented): {obs}")

    # Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step: obs={obs}, reward={reward:.1f}")
        if terminated or truncated:
            break


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TinderForRL: New Features Examples")
    print("=" * 70)

    print("\nYou can run individual examples or all together.")
    print("This script demonstrates:")
    print("  1. Tile Coding training with TensorBoard")
    print("  2. Energy augmented observations")
    print("  3. Policy interpretability via decision trees")
    print("  4. Multi-seed evaluation with statistics")
    print("  5. Combined augmentation wrapper")

    # Run examples
    example_1_tile_coding_training()
    example_2_energy_augmented_observation()
    example_3_policy_interpretability()
    example_4_multiseed_evaluation()
    example_5_combined_augmentation()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
