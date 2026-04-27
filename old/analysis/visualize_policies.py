"""
Policy visualization script for all RL approaches.

Generates policy heatmaps showing:
- Q-table policies (discrete action choices as heatmaps)
- Deep RL policies (action magnitudes across state space)
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import importlib.util
from stable_baselines3 import TD3, SAC

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load QTableAgent
AGENT_PATH = os.path.join(CURRENT_DIR, "..", "agents", "agent_qtable.py")
spec = importlib.util.spec_from_file_location("agent_qtable", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
QTableAgent = agent_mod.QTableAgent


def visualize_discrete_qtable_policy():
    """Visualize discrete Q-table policy as heatmap."""
    print("Generating discrete Q-table policy visualization...")
    
    # Load Q-table
    q_table_path = os.path.join(
        CURRENT_DIR, "..", "results", "models", "q_table.npy"
    )
    q_table = np.load(q_table_path)
    
    env = gym.make("MountainCar-v0")
    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=[48, 48],
        num_actions=env.action_space.n,
    )
    agent.q_table = q_table
    agent.epsilon = 0.0
    
    # Generate policy grid
    pos_bins = np.linspace(env.observation_space.low[0], 
                           env.observation_space.high[0], 48)
    vel_bins = np.linspace(env.observation_space.low[1], 
                           env.observation_space.high[1], 48)
    
    policy_grid = np.zeros((len(vel_bins), len(pos_bins)))
    q_max_grid = np.zeros((len(vel_bins), len(pos_bins)))
    
    for i, vel in enumerate(vel_bins):
        for j, pos in enumerate(pos_bins):
            state = agent.discretize_state(np.array([pos, vel]))
            policy_grid[i, j] = np.argmax(agent.q_table[state])
            q_max_grid[i, j] = np.max(agent.q_table[state])
    
    # Plot 1: Policy (action choices)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im0 = axes[0].imshow(policy_grid, cmap='cool', origin='lower', 
                         extent=[pos_bins[0], pos_bins[-1], 
                                vel_bins[0], vel_bins[-1]],
                         aspect='auto')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title('Discrete Q-Table Policy\n(0=Left, 1=Neutral, 2=Right)')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('Action')
    
    # Plot 2: Max Q-values
    im1 = axes[1].imshow(q_max_grid, cmap='hot', origin='lower',
                         extent=[pos_bins[0], pos_bins[-1], 
                                vel_bins[0], vel_bins[-1]],
                         aspect='auto')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Maximum Q-Values by State\n(Brighter = Higher Confidence)')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label('Max Q-Value')
    
    plt.tight_layout()
    
    plot_dir = os.path.join(CURRENT_DIR, "..", "results", "policies")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "discrete_qtable_policy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def visualize_continuous_qtable_policy():
    """Visualize continuous Q-table policy."""
    print("Generating continuous Q-table policy visualization...")
    
    q_table_path = os.path.join(
        CURRENT_DIR, "..", "results", "models", "continuous_q_table.npy"
    )
    q_table = np.load(q_table_path)
    
    env = gym.make("MountainCar-v0")
    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=[100, 100],
        num_actions=env.action_space.n,
    )
    agent.q_table = q_table
    agent.epsilon = 0.0
    
    # Generate policy grid
    pos_bins = np.linspace(env.observation_space.low[0], 
                           env.observation_space.high[0], 100)
    vel_bins = np.linspace(env.observation_space.low[1], 
                           env.observation_space.high[1], 100)
    
    policy_grid = np.zeros((len(vel_bins), len(pos_bins)))
    q_max_grid = np.zeros((len(vel_bins), len(pos_bins)))
    
    for i, vel in enumerate(vel_bins):
        for j, pos in enumerate(pos_bins):
            state = agent.discretize_state(np.array([pos, vel]))
            policy_grid[i, j] = np.argmax(agent.q_table[state])
            q_max_grid[i, j] = np.max(agent.q_table[state])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im0 = axes[0].imshow(policy_grid, cmap='cool', origin='lower',
                         extent=[pos_bins[0], pos_bins[-1], 
                                vel_bins[0], vel_bins[-1]],
                         aspect='auto')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title('Continuous Q-Table Policy\n(Non-Null Action Cost)')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('Action')
    
    im1 = axes[1].imshow(q_max_grid, cmap='hot', origin='lower',
                         extent=[pos_bins[0], pos_bins[-1], 
                                vel_bins[0], vel_bins[-1]],
                         aspect='auto')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Maximum Q-Values\n(Action Cost Penalty)')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label('Max Q-Value')
    
    plt.tight_layout()
    
    plot_dir = os.path.join(CURRENT_DIR, "..", "results", "policies")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "continuous_qtable_policy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def visualize_deeprl_policy(model_path, model_type="TD3"):
    """Visualize deep RL policy across state space."""
    print(f"Generating {model_type} policy visualization...")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"  ⚠ Model not found: {model_path}")
        return
    
    # Load model
    env = gym.make("MountainCarContinuous-v0")
    if model_type == "TD3":
        model = TD3.load(model_path, env=env)
    elif model_type == "SAC":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate policy grid
    pos_range = np.linspace(-1.2, 0.6, 50)
    vel_range = np.linspace(-0.07, 0.07, 50)
    
    action_magnitude = np.zeros((len(vel_range), len(pos_range)))
    action_direction = np.zeros((len(vel_range), len(pos_range)))
    
    for i, vel in enumerate(vel_range):
        for j, pos in enumerate(pos_range):
            obs = np.array([pos, vel], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            action_magnitude[i, j] = np.abs(action[0])
            action_direction[i, j] = action[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Action magnitude
    im0 = axes[0].imshow(action_magnitude, cmap='Reds', origin='lower',
                         extent=[pos_range[0], pos_range[-1], 
                                vel_range[0], vel_range[-1]],
                         aspect='auto')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title(f'{model_type} Action Magnitude\n(Brighter = Stronger Force)')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('|Action|')
    
    # Action direction
    im1 = axes[1].imshow(action_direction, cmap='RdBu_r', origin='lower',
                         extent=[pos_range[0], pos_range[-1], 
                                vel_range[0], vel_range[-1]],
                         aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title(f'{model_type} Action Direction\n(Red=Left, Blue=Right)')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label('Action Value')
    
    plt.tight_layout()
    
    plot_dir = os.path.join(CURRENT_DIR, "..", "results", "policies")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{model_type.lower()}_policy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def main():
    """Generate all policy visualizations."""
    print("\n" + "=" * 70)
    print("GENERATING POLICY VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    # Visualize Q-table policies
    try:
        visualize_discrete_qtable_policy()
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    try:
        visualize_continuous_qtable_policy()
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    # Visualize deep RL policies
    td3_path = os.path.join(CURRENT_DIR, "..", "results", "models", "td3_continuous_intensity")
    sac_path = os.path.join(CURRENT_DIR, "..", "results", "models", "sac_continuous_intensity")
    
    try:
        visualize_deeprl_policy(td3_path, "TD3")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    try:
        visualize_deeprl_policy(sac_path, "SAC")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Policy visualizations complete!")
    print("Check results/policies/ for generated heatmaps")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
