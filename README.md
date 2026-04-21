# TinderForRL: Matching RL Algorithms to Problems

A comprehensive study of reinforcement learning algorithms on the Mountain Car environment.
The goal: understand why different algorithms excel under different reward structures.

## Project Structure

```
TinderForRL/
├── training/                          # Training scripts
│   ├── train_qtable_discrete.py      # Discrete Q-learning with velocity shaping
│   ├── train_continuous_qtable.py    # Continuous Q-learning with action cost penalty
│   └── train_continuous_deeprl.py    # TD3 & SAC with action intensity cost
│
├── agents/                            # RL agent implementations
│   └── agent_qtable.py               # Q-table agent with tile coding
│
├── config/                            # Training configurations
│   ├── qtable_discrete.yaml          # Discrete Q-learning params
│   ├── qtable_continuous.yaml        # Continuous Q-learning params
│   └── deeprl.yaml                   # Deep RL (TD3/SAC) params
│
├── analysis/                          # Analysis notebooks & scripts
│   ├── qtable_discrete_analysis.ipynb      # Discrete results analysis
│   ├── qtable_continuous_analysis.ipynb    # Continuous results analysis
│   └── compare_all_approaches.py           # Cross-algorithm comparison
│
├── results/                           # Training outputs
│   ├── models/                        # Saved trained models
│   ├── metrics/                       # Training metrics (rewards, etc)
│   └── comparison/                    # Comparison plots
│
├── requirements.txt                   # Python dependencies
├── run_all_training.py               # Master training orchestrator
└── generate_continuous_plots.py      # Visualization utilities
```

## Algorithms Implemented

### 1. **Discrete Q-Learning with Tile Coding** (Discrete-v0)
- **Approach**: Discretize continuous state space into bins, then tabular Q-learning
- **Key Feature**: Velocity-based reward shaping
- **Why**: Simple, interpretable, shows the Q-function as a 2D heatmap
- **Strengths**: Fast training, clear policy visualization
- **Weaknesses**: State discretization is a design choice; can't scale to high dimensions

### 2. **Continuous Q-Learning with Action Cost** (Discrete-v0 + wrapper)
- **Approach**: Same Q-table, but add cost for non-neutral actions
- **Reward**: r = -1/step - 0.1 × (1 if action ≠ neutral else 0)
- **Why**: Tests how reward shaping affects learned policy
- **Strengths**: Shows exploration-exploitation tradeoff
- **Weaknesses**: Action discretization (only 3 actions)

### 3. **TD3 with Action Intensity Cost** (Continuous-v0)
- **Approach**: Deep deterministic policy gradient with twin networks
- **Reward**: r = -0.1 × action² (standard continuous control penalty)
- **Why**: Handles continuous action spaces natively
- **Strengths**: Stable, sample-efficient, learns pure continuous policies
- **Weaknesses**: Black box, harder to interpret than Q-tables

### 4. **SAC with Action Intensity Cost** (Continuous-v0)
- **Approach**: Soft actor-critic with entropy regularization
- **Reward**: r = -0.1 × action² (same as TD3)
- **Why**: More stable than TD3, auto-tunes exploration
- **Strengths**: Natural entropy bonus encourages exploration
- **Weaknesses**: Stochastic policy makes evaluation harder

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run all training
```bash
python run_all_training.py
```

This will:
- Train discrete Q-learning
- Train continuous Q-learning  
- Train TD3 and SAC
- Generate comparison plots
- Print summary metrics

Or run individual scripts:

```bash
# Discrete Q-learning only
python training/train_qtable_discrete.py

# Continuous Q-learning only
python training/train_continuous_qtable.py

# Deep RL only
python training/train_continuous_deeprl.py

# Analysis & comparison
python analysis/compare_all_approaches.py
```

### 3. Analysis
Open the Jupyter notebooks:
```bash
jupyter notebook analysis/qtable_discrete_analysis.ipynb
jupyter notebook analysis/qtable_continuous_analysis.ipynb
```

## Key Results

Run the comparison script to see metrics like:
- **Success Rate**: % of episodes that reach the goal
- **Average Steps**: Convergence speed
- **Action Efficiency**: For continuous: total action cost incurred

## Why This Project Matters

This assignment teaches:

1. **Algorithm Selection is Critical**
   - Different state/action spaces demand different approaches
   - Discrete states → Q-table (simple, fast)
   - Continuous states → Neural nets or discretization
   - Discrete actions → Value-based (DQN, Q-learning)
   - Continuous actions → Policy gradients or Actor-Critic (TD3, SAC, PPO)

2. **Reward Shaping is an Art**
   - Same environment, different reward → different behaviors
   - Velocity shaping vs position shaping vs action cost
   - Trade-offs: exploration speed vs sample efficiency

3. **Interpretability vs Power**
   - Q-tables let you visualize the entire learned policy
   - Deep RL scales but becomes opaque
   - No one-size-fits-all solution

## Next Steps

- [ ] Implement policy visualization for deep RL agents
- [ ] Add ablation studies (remove shaping, change hyperparameters)
- [ ] Implement PPO for continuous as alternative to TD3/SAC
- [ ] Part 02: Find and analyze impressive RL paper

## References

- Sutton & Barto, Reinforcement Learning: An Introduction
- OpenAI Spinning Up: [https://spinningup.openai.com](https://spinningup.openai.com)
- TD3 Paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- SAC Paper: [Soft Actor-Critic: Off-Policy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)