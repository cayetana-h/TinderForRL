"""

# New Features Guide: TinderForRL Enhancements

This document explains the five new features added to the TinderForRL project.

---

## 1. TensorBoard Logging in Training Loop

### What it does
Logs training metrics (reward, steps, epsilon, loss) to TensorBoard for real-time visualization.

### Files Modified
- `training/train_qtable_discrete.py` - Now includes SummaryWriter

### How to use it

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer before training loop
writer = SummaryWriter(log_dir="runs/my_experiment")

# Log metrics during training
for episode in range(num_episodes):
    # ... training code ...
    writer.add_scalar("Reward/raw", total_reward, episode)
    writer.add_scalar("Reward/shaped", shaped_reward, episode)
    writer.add_scalar("Metrics/steps", steps, episode)
    writer.add_scalar("Metrics/epsilon", epsilon, episode)

# Close writer
writer.close()

# View in TensorBoard
# tensorboard --logdir runs/
```

### Output
- Logs saved to: `runs/qtable_discrete/`
- Scalars tracked:
  - `Reward/raw`: Raw (unshapen) episode reward
  - `Reward/shaped`: Reward after shaping
  - `Metrics/steps`: Steps taken in episode
  - `Metrics/epsilon`: Exploration rate
  - `Evaluation/greedy_wins`: Successes on greedy evaluation

---

## 2. Tile Coding / Linear Function Approximation Agent

### What it does
Uses tile coding to discretize continuous states, then applies linear Q-learning.
More scalable than tabular Q-learning, can generalize to state regions not visited during training.

### Files Added
- `agents/agent_tilecoding.py` - TileCoder and TileCodingQAgent classes
- `training/train_tilecoding_discrete.py` - Training script
- `config/tilecoding_discrete.yaml` - Configuration

### How to use it

```python
from agents.agent_tilecoding import TileCodingQAgent

# Create agent
agent = TileCodingQAgent(
    state_low=[-1.2, -0.07],
    state_high=[0.6, 0.07],
    num_actions=3,
    num_tilings=8,        # Number of overlapping tile grids
    tiles_per_dim=[8, 8], # Tiles per dimension
    alpha=0.01,           # Learning rate
    gamma=0.99,
    epsilon_start=1.0,
)

# Training loop
for episode in range(num_episodes):
    obs, _ = env.reset()
    for step in range(max_steps):
        action = agent.select_action(obs, training=True)
        obs_next, reward, done, _ = env.step(action)
        agent.update(obs, action, reward, obs_next, done)
        obs = obs_next

# Evaluation
action = agent.select_action(obs, training=False)

# Save/load
agent.save("weights.npy")
agent.load("weights.npy")
```

### Key Features
- **Tile Coding**: Creates multiple overlapping grids (tilings) offset from each other
- **Feature Size**: num_tilings × (tiles_per_dim[0] × tiles_per_dim[1])
- **Generalization**: Smoother Q-function than tabular approach
- **TensorBoard Integration**: Already included in training script

### Training
```bash
python training/train_tilecoding_discrete.py
# Outputs: results/models/tilecoding_weights.npy
# Logs: runs/tilecoding_discrete/
```

---

## 3. Augmented Observation Wrapper (Energy Features)

### What it does
Augments MountainCar observations with kinetic and potential energy, giving agents access to 
physically meaningful features without manual engineering.

### Files Added
- `utils/wrappers.py` - Contains EnergyAugmentWrapper, RewardShapingWrapper, CombinedAugmentationWrapper

### How to use it

```python
from utils.wrappers import EnergyAugmentWrapper
import gymnasium as gym

# Method 1: Energy augmentation only
env = gym.make("MountainCar-v0")
env = EnergyAugmentWrapper(env)

obs, _ = env.reset()
# Original: [position, velocity]
# Augmented: [position, velocity, kinetic_energy, potential_energy]
print(obs.shape)  # (4,)
print(obs)  # [pos, vel, KE, PE]

# Method 2: Combined augmentation wrapper
from utils.wrappers import CombinedAugmentationWrapper
env = gym.make("MountainCar-v0")
env = CombinedAugmentationWrapper(env, add_energy=True)
```

### Observation Format
- `obs[0]`: Position (-1.2 to 0.6)
- `obs[1]`: Velocity (-0.07 to 0.07)
- `obs[2]`: Kinetic energy (0.0 to 1.0, normalized)
- `obs[3]`: Potential energy (-1.0 to 1.0, normalized)

### Physics
- KE = 0.5 × mass × velocity²
- PE = mass × gravity × height
- Height = position - min_position

### Training with Augmented Observations
Create an agent that accepts 4-D state:

```python
from agents.agent_tilecoding import TileCodingQAgent

env = EnergyAugmentWrapper(gym.make("MountainCar-v0"))

agent = TileCodingQAgent(
    state_low=env.observation_space.low,      # [-1.2, -0.07, 0.0, -1.0]
    state_high=env.observation_space.high,    # [0.6, 0.07, 1.0, 1.0]
    num_actions=3,
    num_tilings=8,
    tiles_per_dim=[8, 8, 4, 4],  # More tiles for more features
    alpha=0.001,
)
```

---

## 4. Policy Interpretability Script

### What it does
Extracts the learned policy from a Q-table and fits a DecisionTreeClassifier to explain
which state features (position, velocity) drive action selection.

### Files Added
- `analysis/qtable_interpretability.py` - Complete interpretability pipeline

### How to use it

```bash
# CLI usage
python analysis/qtable_interpretability.py

# Python usage
from analysis.qtable_interpretability import analyze_qtable_policy
import gymnasium as gym

env = gym.make("MountainCar-v0")
q_table_path = "results/models/q_table.npy"
num_bins = [200, 200]
env_bounds = {"low": env.observation_space.low, "high": env.observation_space.high}

clf, states, actions = analyze_qtable_policy(
    q_table_path,
    num_bins,
    env_bounds,
    output_dir="results/interpretability",
    max_tree_depth=4
)
```

### Outputs
1. **Policy Tree Structure** (`policy_tree.png`)
   - Visual representation of decision tree
   - Shows splits on position and velocity
   - Leaves show predicted actions (left/neutral/right)

2. **Feature Importances** (`feature_importances.png`)
   - Bar plot showing how important each feature is
   - Higher bar = more important for driving policy

3. **Console Summary**
   ```
   Feature Importances:
     position    : 0.7234
     velocity    : 0.2766
   
   Tree Depth: 4
   Num Leaves: 8
   Policy Complexity Score: 0.08
   ```

### Interpretation Examples

**High position importance, low velocity:**
- Policy primarily depends on where the car is
- Actions: "go left if too far left, go right if too far right"

**High velocity importance:**
- Policy depends on momentum
- Actions: "consider building speed, then push harder"

**Balanced importance:**
- Both position and velocity matter equally
- Policy considers both location and momentum

---

## 5. Multi-Seed Evaluation Loop

### What it does
Runs trained agents across N=10 random seeds, computing robust statistics (mean, std, 95% CI)
for episode reward and steps-to-goal.

### Files Added
- `evaluation/multi_seed_eval.py` - MultiSeedEvaluator class and convenience functions

### How to use it

```python
from evaluation.multi_seed_eval import (
    MultiSeedEvaluator,
    evaluate_qtable_agent,
    evaluate_tilecoding_agent
)

# Method 1: Simple evaluation (Q-table)
results = evaluate_qtable_agent(
    q_table_path="results/models/q_table.npy",
    num_bins=[200, 200],
    num_seeds=10  # Run 10 random seeds
)

# Method 2: Custom agent evaluation
evaluator = MultiSeedEvaluator("MountainCar-v0", num_seeds=10, episodes_per_seed=100)

def load_my_agent(path, seed):
    agent = MyAgent(...)
    agent.load(path)
    agent.epsilon = 0.0  # Greedy during evaluation
    return agent

results = evaluator.evaluate(load_my_agent, agent_path="path/to/weights")
```

### Output Statistics

```
======================================================================
EVALUATION RESULTS: Q-Table Agent
======================================================================

Episode Reward (per seed average):
  Mean:           -95.34
  Std:            12.45
  95% CI:   [-108.32,  -82.36]

Steps to Goal (per seed average):
  Mean:           165.23
  Std:            18.67
  95% CI:   [143.45,  187.01]

Success Rate (seeds):
  Mean:           100.00%
  Std:             0.00%
```

### Comparison Between Agents

```python
results1 = evaluate_qtable_agent("results/models/q_table.npy", num_bins=[200, 200])
results2 = evaluate_tilecoding_agent("results/models/tilecoding_weights.npy")

MultiSeedEvaluator.compare_agents(
    {"Q-Table": results1, "Tile Coding": results2},
    metric="reward"
)

# Output:
# 1. Q-Table            :    -95.34 ±   12.45 [95% CI:  -108.32,   -82.36]
# 2. Tile Coding        :    -98.12 ±   15.23 [95% CI:  -115.67,   -80.57]
```

### Statistics Explained

- **Mean**: Average episode reward across all seeds
- **Std**: Standard deviation (variability)
- **95% CI**: 95% confidence interval using t-distribution
  - If CI is [-100, -90], we're 95% confident true mean is in this range
  - Narrower CI = more consistent performance

---

## Integration Example: Complete Workflow

```python
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from agents.agent_tilecoding import TileCodingQAgent
from utils.wrappers import EnergyAugmentWrapper
from analysis.qtable_interpretability import analyze_qtable_policy
from evaluation.multi_seed_eval import evaluate_qtable_agent

# 1. Create augmented environment
env = gym.make("MountainCar-v0")
env = EnergyAugmentWrapper(env)

# 2. Create agent
agent = TileCodingQAgent(
    state_low=env.observation_space.low,
    state_high=env.observation_space.high,
    num_actions=env.action_space.n,
    num_tilings=8,
    tiles_per_dim=[8, 8, 4, 4],
)

# 3. Train with TensorBoard logging
writer = SummaryWriter(log_dir="runs/experiment_1")

for episode in range(5000):
    obs, _ = env.reset()
    total_reward = 0
    for step in range(500):
        action = agent.select_action(obs, training=True)
        obs_next, reward, done, _ = env.step(action)
        agent.update(obs, action, reward, obs_next, done)
        obs = obs_next
        total_reward += reward
        if done:
            break
    writer.add_scalar("Reward", total_reward, episode)

writer.close()
agent.save("results/models/my_agent.npy")

# 4. Multi-seed evaluation
results = evaluate_qtable_agent(
    q_table_path="results/models/q_table.npy",
    num_bins=[200, 200],
    num_seeds=10
)

# 5. Policy analysis (if using tabular Q-learning)
clf, _, _ = analyze_qtable_policy(
    "results/models/q_table.npy",
    [200, 200],
    {"low": env.observation_space.low[:2], "high": env.observation_space.high[:2]}
)
```

---

## Quick Reference

| Feature | File | Use | Output |
|---------|------|-----|--------|
| TensorBoard | train_qtable_discrete.py | Monitor training | runs/qtable_discrete/ |
| Tile Coding | agents/agent_tilecoding.py | Scalable tabular RL | tilecoding_weights.npy |
| Energy Wrapper | utils/wrappers.py | Physics-aware obs | 4-D observations |
| Interpretability | analysis/qtable_interpretability.py | Understand policy | PNG plots + summary |
| Multi-Seed Eval | evaluation/multi_seed_eval.py | Robust statistics | Mean ± std + 95% CI |

---

## Running Examples

Run the complete examples:
```bash
python examples_new_features.py
```

This will demonstrate all five features in action.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install PyTorch:
```bash
pip install torch tensorboard scikit-learn scipy
```

### "Path not found" errors
Make sure you train models first:
```bash
python training/train_qtable_discrete.py
python training/train_tilecoding_discrete.py
```

### TensorBoard not starting
Ensure TensorBoard is installed:
```bash
pip install tensorboard
tensorboard --logdir runs/
```

---

## References

- Tile Coding: Sutton & Barto (2018), Chapter 9
- Potential-based Shaping: Ng et al. (1999)
- Decision Trees: Scikit-learn documentation
- TensorBoard: TensorFlow documentation

"""
