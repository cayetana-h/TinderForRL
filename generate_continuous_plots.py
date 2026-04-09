import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
import os

# Use the script directory as the base path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to agent
AGENTS_PATH = os.path.join(CURRENT_DIR, "agents", "agent_qtable.py")

# Load QTableAgent dynamically
_spec = importlib.util.spec_from_file_location("agent_qtable", AGENTS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QTableAgent = _mod.QTableAgent

print("Agent loaded successfully from:", AGENTS_PATH)

# Consistent plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Load the trained Q-table
q_table_path = os.path.join(CURRENT_DIR, "results", "models", "continuous_q_table.npy")
q_table = np.load(q_table_path)

# Load metrics
rewards = np.load(os.path.join(CURRENT_DIR, "results", "metrics", "continuous_rewards.npy"))
action_counts = np.load(os.path.join(CURRENT_DIR, "results", "metrics", "continuous_action_counts.npy"))
costs = np.load(os.path.join(CURRENT_DIR, "results", "metrics", "continuous_costs.npy"))

print(f"Q-table shape: {q_table.shape}")
print(f"Total episodes: {len(rewards)}")
print(f"Max reward: {np.max(rewards):.2f}")
print(f"Min reward: {np.min(rewards):.2f}")
print(f"Average action count: {np.mean(action_counts):.2f}")
print(f"Average cost: {np.mean(costs):.2f}")

# Plot learning curves
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Rewards
axes[0].plot(rewards, alpha=0.7)
axes[0].set_title("Episode Rewards")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total Reward")
axes[0].grid(True)

# Action counts
axes[1].plot(action_counts, alpha=0.7, color='orange')
axes[1].set_title("Non-Null Actions per Episode")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Action Count")
axes[1].grid(True)

# Costs
axes[2].plot(costs, alpha=0.7, color='red')
axes[2].set_title("Total Cost per Episode")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Cost")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_DIR, "results", "learning_curves_continuous.png"), dpi=150, bbox_inches='tight')
plt.close()

# Recreate agent for discretization
env = gym.make("MountainCar-v0")
agent = QTableAgent(
    state_low=env.observation_space.low,
    state_high=env.observation_space.high,
    num_bins=[100, 100],
    num_actions=3,
)
agent.q_table = q_table

# Create policy grid
pos_bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
vel_bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)

policy = np.zeros((100, 100))
for i, pos in enumerate(pos_bins):
    for j, vel in enumerate(vel_bins):
        state = agent.discretize_state([pos, vel])
        policy[i, j] = np.argmax(q_table[state])

# Plot policy
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.cm.get_cmap('viridis', 3)
im = ax.imshow(policy.T, origin='lower', extent=[pos_bins[0], pos_bins[-1], vel_bins[0], vel_bins[-1]], cmap=cmap, aspect='auto')
ax.set_title("Learned Policy")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")

# Colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.set_label("Action")
cbar.set_ticklabels(["Left", "Neutral", "Right"])

plt.savefig(os.path.join(CURRENT_DIR, "results", "policy_visualization_continuous.png"), dpi=150, bbox_inches='tight')
plt.close()

# Value surface
value_surface = np.max(q_table, axis=2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(pos_bins, vel_bins)
ax.plot_surface(X, Y, value_surface.T, cmap='viridis')
ax.set_title("Value Surface")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
ax.set_zlabel("Max Q-Value")

plt.savefig(os.path.join(CURRENT_DIR, "results", "value_surface_continuous.png"), dpi=150, bbox_inches='tight')
plt.close()

print("Plots saved successfully.")