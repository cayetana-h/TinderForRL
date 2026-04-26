"""
Evaluate and compare Q-Learning vs DQN agents on elevator scheduling.

Generates comparison plots and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_results(prefix):
    """Load training results for an algorithm."""
    try:
        rewards = np.load(f"{prefix}_rewards.npy")
        energy = np.load(f"{prefix}_energy.npy")
        deliveries = np.load(f"{prefix}_deliveries.npy")
        return rewards, energy, deliveries
    except FileNotFoundError:
        print(f"Warning: Could not load results for {prefix}")
        return None, None, None


def compute_stats(rewards, energy, deliveries, window=50):
    """Compute key statistics."""
    smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    smooth_energy = np.convolve(energy, np.ones(window)/window, mode='valid')
    smooth_delivered = np.convolve(deliveries, np.ones(window)/window, mode='valid')
    
    stats = {
        'final_avg_reward': np.mean(smooth_rewards[-50:]),
        'final_avg_energy': np.mean(smooth_energy[-50:]),
        'final_avg_delivered': np.mean(smooth_delivered[-50:]),
        'best_reward': np.max(smooth_rewards),
        'best_episode': np.argmax(smooth_rewards),
        'convergence_episodes': np.where(smooth_rewards > np.mean(smooth_rewards[-50:]) * 0.9)[0][0] if len(np.where(smooth_rewards > np.mean(smooth_rewards[-50:]) * 0.9)[0]) > 0 else len(smooth_rewards)
    }
    
    return stats, smooth_rewards, smooth_energy, smooth_delivered


def create_comparison_plots():
    """Create comprehensive comparison plots."""
    
    # Load results
    dqn_r, dqn_e, dqn_d = load_results("dqn")
    ql_r, ql_e, ql_d = load_results("qlearning")
    
    if dqn_r is None or ql_r is None:
        print("Could not load training results. Run train_elevator_dqn.py and train_elevator_qlearning.py first.")
        return
    
    # Compute statistics
    dqn_stats, dqn_sr, dqn_se, dqn_sd = compute_stats(dqn_r, dqn_e, dqn_d)
    ql_stats, ql_sr, ql_se, ql_sd = compute_stats(ql_r, ql_e, ql_d)
    
    print("\n" + "="*70)
    print("ELEVATOR SCHEDULING COMPARISON: DQN vs Q-Learning")
    print("="*70)
    print(f"\n{'Metric':<30} {'DQN':<20} {'Q-Learning':<20}")
    print("-"*70)
    print(f"{'Final Avg Reward':<30} {dqn_stats['final_avg_reward']:>18.2f} {ql_stats['final_avg_reward']:>18.2f}")
    print(f"{'Final Avg Energy':<30} {dqn_stats['final_avg_energy']:>18.1f} {ql_stats['final_avg_energy']:>18.1f}")
    print(f"{'Final Avg Delivered':<30} {dqn_stats['final_avg_delivered']:>18.1f} {ql_stats['final_avg_delivered']:>18.1f}")
    print(f"{'Best Episode Reward':<30} {dqn_stats['best_reward']:>18.2f} {ql_stats['best_reward']:>18.2f}")
    print(f"{'Convergence Episod':<30} {dqn_stats['convergence_episodes']:>18d} {ql_stats['convergence_episodes']:>18d}")
    print("="*70)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # Row 1: Raw metrics
    x_range_dqn = range(len(dqn_r))
    x_range_ql = range(len(ql_r))
    
    axes[0, 0].plot(dqn_r, alpha=0.3, label='DQN (raw)', color='blue')
    axes[0, 0].plot(dqn_sr, label='DQN (smoothed)', color='darkblue', linewidth=2)
    axes[0, 0].plot(ql_r, alpha=0.3, label='Q-Learning (raw)', color='orange')
    axes[0, 0].plot(ql_sr, label='Q-Learning (smoothed)', color='darkorange', linewidth=2)
    axes[0, 0].set_title("Episode Reward Comparison", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(dqn_se, label='DQN', color='darkblue', linewidth=2)
    axes[0, 1].plot(ql_se, label='Q-Learning', color='darkorange', linewidth=2)
    axes[0, 1].set_title("Energy Consumption Comparison", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Energy Used")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(dqn_sd, label='DQN', color='darkblue', linewidth=2)
    axes[0, 2].plot(ql_sd, label='Q-Learning', color='darkorange', linewidth=2)
    axes[0, 2].set_title("Passengers Delivered Comparison", fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Statistical comparison
    categories = ['Final Reward', 'Final Energy', 'Final Delivered']
    dqn_values = [dqn_stats['final_avg_reward'], -dqn_stats['final_avg_energy'], dqn_stats['final_avg_delivered']]
    ql_values = [ql_stats['final_avg_reward'], -ql_stats['final_avg_energy'], ql_stats['final_avg_delivered']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, dqn_values, width, label='DQN', color='darkblue')
    axes[1, 0].bar(x + width/2, ql_values, width, label='Q-Learning', color='darkorange')
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Final Performance Metrics", fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(categories, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Convergence speed (where does it reach 80% of final performance?)
    dqn_threshold = dqn_stats['final_avg_reward'] * 0.8
    ql_threshold = ql_stats['final_avg_reward'] * 0.8
    
    dqn_convergence = np.where(dqn_sr > dqn_threshold)[0][0] if len(np.where(dqn_sr > dqn_threshold)[0]) > 0 else len(dqn_sr)
    ql_convergence = np.where(ql_sr > ql_threshold)[0][0] if len(np.where(ql_sr > ql_threshold)[0]) > 0 else len(ql_sr)
    
    algorithms = ['DQN', 'Q-Learning']
    convergence_times = [dqn_convergence, ql_convergence]
    
    axes[1, 1].bar(algorithms, convergence_times, color=['darkblue', 'darkorange'])
    axes[1, 1].set_ylabel("Episodes")
    axes[1, 1].set_title("Convergence Speed (80% of Final)", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(convergence_times):
        axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Efficiency comparison (reward per energy)
    dqn_efficiency = dqn_stats['final_avg_reward'] / max(dqn_stats['final_avg_energy'], 1)
    ql_efficiency = ql_stats['final_avg_reward'] / max(ql_stats['final_avg_energy'], 1)
    
    axes[1, 2].bar(algorithms, [dqn_efficiency, ql_efficiency], color=['darkblue', 'darkorange'])
    axes[1, 2].set_ylabel("Reward / Energy")
    axes[1, 2].set_title("Efficiency: Reward per Energy Unit", fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([dqn_efficiency, ql_efficiency]):
        axes[1, 2].text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("elevator_comparison.png", dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved as elevator_comparison.png")
    
    return dqn_stats, ql_stats


if __name__ == "__main__":
    create_comparison_plots()
