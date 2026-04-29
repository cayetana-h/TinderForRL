# TinderForRL Visualization & Analysis Tools

Complete suite of Jupyter notebooks for comprehensive RL analysis covering all aspects of your research: state representations, rewards, algorithms, training strategies, policy analysis, and explanations.

## 🎯 START HERE: Master Dashboard

### **master_dashboard.ipynb** ⭐ PRIMARY ENTRY POINT
All-in-one comprehensive analysis dashboard covering your entire design checklist.

**Complete Coverage:**
- ✅ **Tensor Log Processing** - Load and visualize TensorBoard metrics
- ✅ **Training Dynamics** - Plot curves, convergence, learning stability
- ✅ **State Representations** - Analyze dimensionality, distributions, PCA
- ✅ **Reward Analysis** - Compare objective vs engineered rewards
- ✅ **Policy Structure** - Extract decision mappings, action distributions
- ✅ **Statistical Comparison** - Confidence intervals, hypothesis testing
- ✅ **Feature Importance** - ML-based policy explanations
- ✅ **Cross-Scenario Comparison** - Algorithm performance across environments
- ✅ **Interactive Visualizations** - Explore and drill-down into details

**Available Functions:**
```python
# Training & Rewards (Scenarios 1-4)
plot_training_curves(scenario=1)                          # Training curves visualization
compare_reward_signals(scenario=1)                        # Objective vs engineered rewards

# State & Environment Analysis
analyze_state_representations(scenario=1)                 # State distributions & PCA

# Policy Analysis
analyze_policy_behavior(scenario=1, algorithm='DQN')      # Action distributions
explain_policy_feature_importance(scenario=1, algorithm='DQN')  # ML explanations

# Statistical Comparison
compare_algorithms_scenario(scenario=1)                   # Statistical metrics
compare_across_scenarios(algorithm='DQN')                 # Cross-scenario performance
```

---

## Specialized Notebooks Overview

### 1. **monitoring_tools.ipynb**
Covers TensorBoard setup and metrics tracking for RL training.

**Key Features:**
- TensorBoard configuration for Stable Baselines3
- Custom callbacks for episode metrics tracking
- Evaluation callbacks for periodic model validation
- TensorBoard log parsing and visualization
- Scenario results monitoring

**When to Use:**
- Setting up training monitoring infrastructure
- Real-time tracking of training progress
- Analyzing training curves and metrics
- Comparing multiple training runs

---

### 2. **evaluation_analysis.ipynb**
Comprehensive evaluation and performance analysis of trained agents.

**Key Features:**
- Load trained models from scenarios
- Evaluate agents deterministically and stochastically
- Compare objective performance vs. engineered rewards
- Statistical analysis (mean, std, min, max rewards)
- Multi-algorithm comparison on same environment
- Reward signal comparison visualization

**When to Use:**
- Evaluating final trained models
- Comparing performance across algorithms
- Analyzing reward signals
- Statistical significance testing
- Creating evaluation reports

---

### 3. **policy_analysis.ipynb**
Structural, statistical, and topological analysis of learned policies.

**Key Features:**
- Policy extraction and action distribution analysis
- **Statistical Analysis:**
  - Entropy (policy diversity/randomness)
  - Action bias detection
  - Policy variance
  - Determinism analysis

- **Structural Analysis:**
  - Neural network structure inspection
  - Weight and activation analysis
  - Parameter configuration

- **Topological Analysis:**
  - 2D policy heatmaps (for 2D state spaces)
  - Decision boundary visualization
  - Policy continuity analysis

- **Comparative Analysis:**
  - Policy similarity metrics
  - Action distribution comparison
  - Cross-algorithm policy comparison
  - Similarity matrix visualization

**When to Use:**
- Understanding learned behaviors
- Comparing policies qualitatively
- Detecting policy biases
- Analyzing policy complexity
- Generating policy topology visualizations

---

### 4. **explanation_tools.ipynb**
ML-based explanation techniques for policy behavior.

**Key Features:**
- State-Action-Value data collection
- **Linear Regression Analysis:**
  - Linear value function approximation
  - Coefficient-based feature importance
  - Interpretable baseline model

- **Non-linear Analysis:**
  - Random Forest value function approximation
  - Tree-based feature importance (MDI)
  - Improved non-linear modeling

- **Permutation Importance:**
  - Feature shuffling analysis
  - Performance drop measurement
  - Uncertainty quantification (mean ± std)

- **Action Selection Analysis:**
  - Classification-based action prediction
  - Which state features drive actions
  - Action prediction accuracy

- **Integrated Pipeline:**
  - Complete policy explanation with all methods
  - Comparative analysis across techniques

**When to Use:**
- Understanding feature importance for value/action
- Interpreting policy decisions
- Identifying critical state dimensions
- Explaining model behavior to stakeholders
- Debugging policy issues

---

### 5. **analysis.ipynb** (Original)
General RL concepts and initial monitoring setup.

---

## Usage Guide

### Setup
Before running these notebooks, ensure you have:
1. Installed all dependencies: `pip install -r requirements.txt`
2. Trained models in the `training/` folder
3. Results saved in the expected structure (e.g., `results/scenario1/models/...`)

### Typical Workflows

#### Monitoring Training (Real-time)
```
1. Run monitoring_tools.ipynb
2. Configure TensorBoard in your training script
3. Use custom callbacks during model.learn()
4. View TensorBoard: tensorboard --logdir ./tensorboard_logs
```

#### Post-Training Evaluation
```
1. Load trained models using evaluation_analysis.ipynb
2. Evaluate on test environments
3. Compare algorithms
4. Generate performance statistics
```

#### Understanding Learned Policies
```
1. Use policy_analysis.ipynb to extract policy properties
2. Analyze entropy, bias, and variance
3. Visualize topology if applicable
4. Compare with other policies
```

#### Explaining Policy Decisions
```
1. Use explanation_tools.ipynb
2. Run comprehensive explanation pipeline
3. Examine feature importance from multiple angles
4. Identify critical state dimensions
```

---

## 🚀 Quick Start Guide

### Step 1: Open Master Dashboard
Open `master_dashboard.ipynb` and run the setup cell to load all dependencies and explore available functions.

### Step 2: Choose Your Analysis Path

**Path A: Complete Scenario Analysis**
```python
scenario = 1
plot_training_curves(scenario)              # See training progress
analyze_state_representations(scenario)     # Understand state space
compare_algorithms_scenario(scenario)       # Compare algorithms statistically
compare_reward_signals(scenario)            # Analyze reward signals
```

**Path B: Deep Dive into One Algorithm**
```python
analyze_policy_behavior(scenario=1, algorithm='DQN')
explain_policy_feature_importance(scenario=1, algorithm='DQN')
```

**Path C: Cross-Scenario Algorithm Comparison**
```python
compare_across_scenarios(algorithm='DQN')   # See performance across all 4 environments
```

### Step 3: Drill Down as Needed
- Use specialized notebooks (`evaluation_analysis.ipynb`, `policy_analysis.ipynb`, etc.) for deeper investigation
- All notebooks integrate with master dashboard outputs

---

## 📊 Data Flow & Integration

```
Training Scripts (training/*.py)
        ↓
    Generates:
    - Metrics JSON files (results/scenario*/metrics/)
    - Trained models (results/scenario*/models/)
        ↓
Master Dashboard loads data and provides:
├─ Training visualization & TensorBoard integration
├─ State space analysis
├─ Reward signal comparison
├─ Policy behavior analysis
├─ ML-based explanations
├─ Statistical comparisons
└─ Cross-scenario analysis
        ↓
Specialized Notebooks can drill deeper:
├─ monitoring_tools.ipynb (TensorBoard details)
```

---


## ✅ Design Checklist Coverage (2026 Update)

All original requirements are now fully addressed, including real-time monitoring and advanced visualization:

### Environmental Design
- [x] **State Representations** - `analyze_state_representations()` with PCA/correlation analysis
- [x] **Reward Signals** - `compare_reward_signals()` with objective vs engineered comparison
- [x] **Custom Wrappers** - Data collection handles environment variations

### Learning & Training
- [x] **RL Algorithms** - Support for DQN, SAC, TD3, Q-Learning
- [x] **Hyperparameters** - Logged in metrics and analyzed via training curves
- [x] **Training Strategies** - Visualized through training dynamics

### Monitoring & Analysis  
- [x] **TensorBoard Integration** - Real-time training monitoring and log visualization
- [x] **Evaluation** - `compare_algorithms_scenario()` with statistical tests
- [x] **Objective vs Engineered Rewards** - `compare_reward_signals()` analysis

### Policy Analysis
- [x] **Statistical** - `compare_algorithms_scenario()` with confidence intervals
- [x] **Structural** - `analyze_policy_behavior()` with network analysis
- [x] **Topological** - Action distributions and decision mapping
- [x] **Comparative** - `compare_across_scenarios()` across environments

### Visualization & Explanation
- [x] **Visualization Tools** - Matplotlib, Seaborn, and TensorBoard for all plots and interactive monitoring
- [x] **Explanation Tools** - `explain_policy_feature_importance()` with Random Forest, permutation importance, and regression

---

## 📊 Visualization Stack

- **Matplotlib**: Core plotting for all static visualizations (curves, histograms, bar/box plots, PCA, etc.)
- **Seaborn**: Enhanced plot styling and statistical visualizations
- **TensorBoard**: Interactive, real-time monitoring of RL training and metrics
- **Custom Python functions**: For policy analysis, feature importance, and comparative visualizations

All tools are integrated in the master dashboard for seamless, publication-quality RL analysis.

## 💡 Example Workflows

### Complete Research Report
```python
# Scenario comparative analysis
for scenario in range(1, 5):
    print(f"\n{'='*60}\nSCENARIO {scenario}\n{'='*60}")
    plot_training_curves(scenario)
    analyze_state_representations(scenario)
    compare_reward_signals(scenario)
    stats_df = compare_algorithms_scenario(scenario)
    
    # Deep dive for top algorithm
    for algo in stats_df['Algorithm'].head(2):
        analyze_policy_behavior(scenario, algo)
        explain_policy_feature_importance(scenario, algo)
```

### Algorithm Benchmarking
```python
# Compare DQN across all environments
compare_across_scenarios(algorithm='DQN')

# Then analyze why it performs differently
for scenario in range(1, 5):
    explain_policy_feature_importance(scenario, 'DQN')
```

### Reward Engineering Validation
```python
# Verify engineered rewards are effective
for scenario in range(1, 5):
    print(f"\n=== SCENARIO {scenario} ===")
    compare_reward_signals(scenario)
    # Check if reward shaping helped convergence
    plot_training_curves(scenario)
```

---

## 🔧 Customization

All functions are designed to be modular and composable. Customize by:

1. **Modifying n_episodes** in collection functions for speed/accuracy tradeoff
2. **Changing algorithms** in comparative functions
3. **Adjusting visualization parameters** (colors, smoothing windows, etc.)
4. **Adding new metrics** to analysis functions

---

## 📁 File Structure

```
visualisation/
├── master_dashboard.ipynb          ⭐ START HERE
├── monitoring_tools.ipynb          (TensorBoard details)
├── evaluation_analysis.ipynb       (Performance metrics)
├── policy_analysis.ipynb           (Policy structure)
├── explanation_tools.ipynb         (Feature importance)
├── tensorboard_logs/               (Train logs - auto-populated)
├── metrics/                        (Analysis outputs)
└── README.md                       (This file)
```

---

## 🐛 Troubleshooting

**"No metrics found for scenario X"**
→ Ensure training scripts have completed and saved JSON files to `results/scenario*/metrics/`

**"Model not found"**
→ Check that trained models exist in `results/scenario*/models/`

**"Missing dependencies"**
→ Install: `pip install ipykernel scikit-learn scipy` into your venv

**Plots not showing**
→ Ensure Jupyter notebook kernel is properly configured (venv python)

---

## 📚 References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Scikit-learn](https://scikit-learn.org/)

```

---

## Scenarios Configuration

The notebooks work with TinderForRL's 4 scenarios:

| Scenario | Environment | Objective | Algorithms |
|----------|-------------|-----------|-----------|
| 1 | MountainCar-v0 | Minimum Steps | DQN, SAC, TD3 |
| 2 | CartPole-v1 | Maximum Performance | DQN, SAC, TD3 |
| 3 | Acrobot-v1 | Minimum Steps | DQN, SAC, TD3 |
| 4 | Pendulum-v1 | Energy Efficiency | DQN, SAC, TD3 |

---

## Dependencies

- `stable-baselines3` - RL algorithms
- `gymnasium` - Environments
- `numpy, pandas` - Data manipulation
- `matplotlib, seaborn` - Visualization
- `scikit-learn` - ML models and metrics
- `tensorboard` - Training monitoring
- `scipy` - Statistical analysis

---

## Outputs & Artifacts

Generate by these notebooks:
- **TensorBoard logs:** `./tensorboard_logs/`
- **Evaluation reports:** In-notebook tables and plots
- **Policy visualizations:** Heatmaps, distributions, topologies
- **Explanation analysis:** Feature importance charts
- **Comparison matrices:** Similarity/performance matrices

---

## Tips & Best Practices

1. **Start with evaluation** - Understand your model's performance first
2. **Then analyze policy** - See what it learned qualitatively  
3. **Then explain** - Understand *why* it makes decisions
4. **Compare systematically** - Use notebooks to compare across algorithms/scenarios
5. **Save results** - Export plots and tables for reports

---

## Troubleshooting

- **Models not loading:** Check paths in `RESULTS_DIR`
- **TensorBoard not showing:** Verify `tensorboard_log=` in training code
- **No evaluation results:** Ensure trained models `.zip` files exist
- **Memory issues:** Reduce `n_episodes` or `n_states` parameters

---

## Future Enhancements

- [ ] SHAP values for policy explanation
- [ ] Policy distillation analysis
- [ ] State importance heatmaps
- [ ] Interactive Plotly visualizations
- [ ] Automated report generation
- [ ] Adversarial robustness analysis

---

For questions or issues, refer to the main TinderForRL README or open an issue.
