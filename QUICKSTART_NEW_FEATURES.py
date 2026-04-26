#!/usr/bin/env python3
"""
Quick start guide for new features.

Run this to get started with the new TinderForRL enhancements.
"""

import sys
import os

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_requirements():
    """Check if all dependencies are installed."""
    print_section("Checking Requirements")
    
    required = {
        'gymnasium': 'gymnasium',
        'numpy': 'numpy',
        'torch': 'torch',
        'tensorboard': 'tensorboard',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'yaml': 'PyYAML',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name}: OK")
        except ImportError:
            print(f"✗ {package_name}: MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All requirements satisfied!")
    return True

def show_usage_examples():
    """Show usage examples for each feature."""
    print_section("Quick Start: Feature Examples")
    
    examples = {
        "TensorBoard Logging": """
# View training progress in real-time
tensorboard --logdir runs/
# Open browser: http://localhost:6006
""",
        
        "Train Tile Coding Agent": """
python training/train_tilecoding_discrete.py
# Output: results/models/tilecoding_weights.npy
# Logs: runs/tilecoding_discrete/
""",
        
        "Use Energy Augmented Observations": """
from utils.wrappers import EnergyAugmentWrapper
import gymnasium as gym

env = gym.make("MountainCar-v0")
env = EnergyAugmentWrapper(env)
obs, _ = env.reset()
print(obs.shape)  # (4,) instead of (2,)
""",
        
        "Analyze Policy with Decision Tree": """
python analysis/qtable_interpretability.py
# Outputs:
# - results/interpretability/policy_tree.png
# - results/interpretability/feature_importances.png
""",
        
        "Multi-Seed Evaluation": """
from evaluation.multi_seed_eval import evaluate_qtable_agent

results = evaluate_qtable_agent(
    q_table_path="results/models/q_table.npy",
    num_bins=[200, 200],
    num_seeds=10
)
""",
    }
    
    for name, code in examples.items():
        print(f"\n{name}:")
        print(code)

def show_workflow():
    """Show complete workflow."""
    print_section("Complete Workflow")
    
    workflow = """
Step 1: Train Q-table agent with TensorBoard logging
    $ python training/train_qtable_discrete.py
    → results/models/q_table.npy
    → runs/qtable_discrete/ (TensorBoard logs)

Step 2: Train tile coding agent
    $ python training/train_tilecoding_discrete.py
    → results/models/tilecoding_weights.npy
    → runs/tilecoding_discrete/

Step 3: Analyze learned policy
    $ python analysis/qtable_interpretability.py
    → results/interpretability/policy_tree.png
    → results/interpretability/feature_importances.png

Step 4: Run multi-seed evaluations
    $ python evaluation/multi_seed_eval.py
    → Statistics with 95% confidence intervals

Step 5: View TensorBoard results
    $ tensorboard --logdir runs/
    → Open http://localhost:6006 in browser

Step 6: Run complete example
    $ python examples_new_features.py
    → Demonstrates all features in action
"""
    print(workflow)

def show_next_steps():
    """Show next steps for the assignment."""
    print_section("Next Steps for Assignment")
    
    steps = """
PART 01 - MountainCar Implementation:
  ✓ Discrete Q-Learning with tabular approach
  ✓ Continuous actions with cost penalty
  ✓ TD3 (actor-critic for continuous)
  ✓ Reward shaping (velocity-based potential)
  
  NEW - Now Added:
  ✓ Tile Coding (function approximation)
  ✓ Energy augmented observations
  ✓ TensorBoard monitoring
  ✓ Policy interpretability (decision tree)
  ✓ Multi-seed evaluation (statistics)
  
  TODO:
  - [ ] Implement DQN for discrete actions
  - [ ] Add SAC for continuous actions
  - [ ] Create feature importance analysis
  - [ ] Generate comparative plots
  - [ ] Write presentation document

PART 02 - RL Application Review:
  - [ ] Select real-world RL application
  - [ ] Write problem formulation (states, actions, rewards)
  - [ ] Document methodology and results
  - [ ] Create presentation slides

DELIVERABLES:
  - [ ] Complete Jupyter Notebook (Part 01)
  - [ ] Presentation PDF/PPTX (15-20 minutes)
  - [ ] ZIP file: RLI_22_00 – Group {XY}.zip
  - [ ] Due: April 29, 2026
"""
    print(steps)

def main():
    """Run quick start guide."""
    print("\n" + "=" * 70)
    print("  TinderForRL: New Features Quick Start Guide")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\nFix missing dependencies, then run again.")
        return
    
    # Show examples
    show_usage_examples()
    
    # Show workflow
    show_workflow()
    
    # Show next steps
    show_next_steps()
    
    print_section("Ready to Go!")
    print("""
You now have all the tools to complete your RL assignment!

Try running:
  1. python training/train_tilecoding_discrete.py
  2. python analysis/qtable_interpretability.py
  3. python evaluation/multi_seed_eval.py
  4. python examples_new_features.py

For detailed documentation, see: NEW_FEATURES_GUIDE.md
""")

if __name__ == "__main__":
    main()
