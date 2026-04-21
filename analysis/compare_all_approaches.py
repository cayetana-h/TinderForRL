"""
Comprehensive comparison of all RL approaches on MountainCar environments.

Compares:
1. Discrete Q-learning (velocity shaping)
2. Continuous Q-learning (non-null action cost)
3. TD3 (continuous action intensity cost)
4. SAC (continuous action intensity cost)

Generates metrics: success rate, convergence speed, policy efficiency
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import importlib.util
from stable_baselines3 import TD3, SAC

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load QTableAgent
AGENT_PATH = os.path.join(CURRENT_DIR, "..", "agents", "agent_qtable.py")
spec = importlib.util.spec_from_file_location("agent_qtable", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
QTableAgent = agent_mod.QTableAgent


class ComparisonAnalyzer:
    """Unified comparison across all approaches."""
    
    def __init__(self):
        self.results = {}
    
    def test_discrete_qtable(self, num_episodes=50):
        """Test discrete Q-table agent."""
        print("\n" + "=" * 70)
        print("Testing Discrete Q-Learning")
        print("=" * 70)
        
        env = gym.make("MountainCar-v0")
        
        # Load trained Q-table
        q_table_path = os.path.join(
            CURRENT_DIR, "..", "results", "models", "q_table.npy"
        )
        q_table = np.load(q_table_path)
        
        agent = QTableAgent(
            state_low=env.observation_space.low,
            state_high=env.observation_space.high,
            num_bins=[48, 48],
            num_actions=env.action_space.n,
        )
        agent.q_table = q_table
        agent.epsilon = 0.0  # Greedy
        
        successes = 0
        total_steps = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            state = agent.discretize_state(obs)
            steps = 0
            
            for step in range(200):
                action = agent.select_action(state)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = agent.discretize_state(obs_next)
                steps += 1
                
                if done:
                    if terminated:
                        successes += 1
                    total_steps.append(steps)
                    break
        
        self.results["discrete_qtable"] = {
            "success_rate": successes / num_episodes,
            "avg_steps": np.mean(total_steps) if total_steps else 200,
            "successes": successes,
        }
        
        print(f"Success Rate: {100 * successes / num_episodes:.1f}%")
        print(f"Avg Steps to Goal: {np.mean(total_steps):.1f}" if total_steps else "No successes")
        
        return self.results["discrete_qtable"]
    
    def test_continuous_qtable(self, num_episodes=50):
        """Test continuous Q-table with action cost."""
        print("\n" + "=" * 70)
        print("Testing Continuous Q-Learning (Non-Null Action Cost)")
        print("=" * 70)
        
        env = gym.make("MountainCar-v0")
        
        # Load trained Q-table
        q_table_path = os.path.join(
            CURRENT_DIR, "..", "results", "models", "continuous_q_table.npy"
        )
        q_table = np.load(q_table_path)
        
        agent = QTableAgent(
            state_low=env.observation_space.low,
            state_high=env.observation_space.high,
            num_bins=[100, 100],
            num_actions=env.action_space.n,
        )
        agent.q_table = q_table
        agent.epsilon = 0.0  # Greedy
        
        successes = 0
        total_steps = []
        action_counts = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            state = agent.discretize_state(obs)
            steps = 0
            non_null_actions = 0
            
            for step in range(1000):
                action = agent.select_action(state)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = agent.discretize_state(obs_next)
                steps += 1
                
                if action != 1:  # Assuming 1 is neutral
                    non_null_actions += 1
                
                if done:
                    if terminated:
                        successes += 1
                    total_steps.append(steps)
                    action_counts.append(non_null_actions)
                    break
        
        self.results["continuous_qtable"] = {
            "success_rate": successes / num_episodes,
            "avg_steps": np.mean(total_steps) if total_steps else 1000,
            "avg_actions": np.mean(action_counts) if action_counts else 1000,
            "successes": successes,
        }
        
        print(f"Success Rate: {100 * successes / num_episodes:.1f}%")
        print(f"Avg Steps: {np.mean(total_steps):.1f}" if total_steps else "No successes")
        print(f"Avg Non-Null Actions: {np.mean(action_counts):.1f}" if action_counts else "N/A")
        
        return self.results["continuous_qtable"]
    
    def test_deeprl_model(self, model_path, model_type, num_episodes=50):
        """Test deep RL model (TD3 or SAC)."""
        print("\n" + "=" * 70)
        print(f"Testing {model_type}")
        print("=" * 70)
        
        env = gym.make("MountainCarContinuous-v0")
        
        # Load model
        if model_type == "TD3":
            model = TD3.load(model_path, env=env)
        elif model_type == "SAC":
            model = SAC.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        successes = 0
        total_steps = []
        total_costs = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            steps = 0
            total_cost = 0
            
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Track cost (action intensity)
                action_cost = 0.1 * np.sum(action ** 2)
                total_cost += action_cost
                
                obs = obs_next
                steps += 1
                
                if done:
                    if terminated:
                        successes += 1
                    total_steps.append(steps)
                    total_costs.append(total_cost)
                    break
        
        key = f"{model_type.lower()}_continuous"
        self.results[key] = {
            "success_rate": successes / num_episodes,
            "avg_steps": np.mean(total_steps) if total_steps else 200,
            "avg_cost": np.mean(total_costs) if total_costs else 0,
            "successes": successes,
        }
        
        print(f"Success Rate: {100 * successes / num_episodes:.1f}%")
        print(f"Avg Steps: {np.mean(total_steps):.1f}" if total_steps else "No successes")
        print(f"Avg Action Intensity Cost: {np.mean(total_costs):.4f}" if total_costs else "N/A")
        
        return self.results[key]
    
    def compare_all(self):
        """Run all tests and generate comparison."""
        # Test discrete
        self.test_discrete_qtable(num_episodes=50)
        
        # Test continuous Q-learning
        self.test_continuous_qtable(num_episodes=50)
        
        # Test deep RL models if they exist
        td3_path = os.path.join(CURRENT_DIR, "..", "results", "models", "td3_continuous_intensity")
        sac_path = os.path.join(CURRENT_DIR, "..", "results", "models", "sac_continuous_intensity")
        
        if os.path.exists(td3_path + ".zip"):
            self.test_deeprl_model(td3_path, "TD3", num_episodes=50)
        
        if os.path.exists(sac_path + ".zip"):
            self.test_deeprl_model(sac_path, "SAC", num_episodes=50)
        
        # Print comparison table
        self.print_comparison()
        self.plot_comparison()
    
    def print_comparison(self):
        """Print results comparison table."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        print(f"\n{'Approach':<30} {'Success Rate':<15} {'Avg Steps':<15}")
        print("-" * 70)
        
        for key, metrics in self.results.items():
            sr = f"{100 * metrics['success_rate']:.1f}%"
            steps = f"{metrics['avg_steps']:.1f}"
            
            name = key.replace("_", " ").title()
            print(f"{name:<30} {sr:<15} {steps:<15}")
        
        print("\nDetailed Results:")
        for key, metrics in self.results.items():
            print(f"\n{key}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
    
    def plot_comparison(self):
        """Generate comparison plots."""
        approaches = list(self.results.keys())
        success_rates = [self.results[a]["success_rate"] * 100 for a in approaches]
        avg_steps = [self.results[a]["avg_steps"] for a in approaches]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Success rate
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        axes[0].bar(range(len(approaches)), success_rates, color=colors[:len(approaches)])
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Success Rate Comparison")
        axes[0].set_xticks(range(len(approaches)))
        axes[0].set_xticklabels([a.replace("_", "\n") for a in approaches], fontsize=9)
        axes[0].set_ylim([0, 105])
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
        
        # Average steps
        axes[1].bar(range(len(approaches)), avg_steps, color=colors[:len(approaches)])
        axes[1].set_ylabel("Average Steps to Goal")
        axes[1].set_title("Convergence Speed Comparison")
        axes[1].set_xticks(range(len(approaches)))
        axes[1].set_xticklabels([a.replace("_", "\n") for a in approaches], fontsize=9)
        for i, v in enumerate(avg_steps):
            axes[1].text(i, v + 2, f"{v:.1f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        
        plot_dir = os.path.join(CURRENT_DIR, "..", "results", "comparison")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "all_approaches_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    analyzer = ComparisonAnalyzer()
    analyzer.compare_all()
