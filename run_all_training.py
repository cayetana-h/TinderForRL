#!/usr/bin/env python3
"""
Master training orchestrator for TinderForRL project.

Runs all RL approaches in sequence:
1. Discrete Q-Learning (velocity shaping)
2. Continuous Q-Learning (non-null action cost)
3. TD3 & SAC (action intensity cost)

Then generates comprehensive analysis and comparison.
"""

import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR  # This is the project root


def run_command(cmd, description):
    """Run a command and report status."""
    print("\n" + "=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"\n⚠ WARNING: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✓ {description} completed successfully")
        return True


def main():
    """Run the full training pipeline."""
    
    print("\n" + "=" * 70)
    print("TINDER FOR RL - FULL TRAINING PIPELINE")
    print("=" * 70)
    print("\nThis will train:")
    print("  1. Discrete Q-Learning (velocity shaping)")
    print("  2. Continuous Q-Learning (non-null action cost)")
    print("  3. TD3 & SAC (action intensity cost)")
    print("\nThen compare all approaches.\n")
    
    # Step 1: Discrete Q-learning
    success = run_command(
        [sys.executable, "training/train_qtable_discrete.py"],
        "Discrete Q-Learning Training"
    )
    if not success:
        print("Failed at step 1. Continuing anyway...")
    
    # Step 2: Continuous Q-learning
    success = run_command(
        [sys.executable, "training/train_continuous_qtable.py"],
        "Continuous Q-Learning Training (Non-Null Action Cost)"
    )
    if not success:
        print("Failed at step 2. Continuing anyway...")
    
    # Step 3: Deep RL (TD3 & SAC)
    success = run_command(
        [sys.executable, "training/train_continuous_deeprl.py"],
        "Deep RL Training (TD3 & SAC)"
    )
    if not success:
        print("Failed at step 3. Continuing anyway...")
    
    # Step 4: Generate plots for discrete
    success = run_command(
        [sys.executable, "generate_continuous_plots.py"],
        "Generate Continuous Plots"
    )
    if not success:
        print("Warning: plot generation failed")
    
    # Step 5: Comparison analysis
    success = run_command(
        [sys.executable, "analysis/compare_all_approaches.py"],
        "Comprehensive Comparison Analysis"
    )
    if not success:
        print("Warning: comparison failed")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - results/models/     (trained models)")
    print("  - results/metrics/    (training metrics)")
    print("  - results/comparison/ (comparison plots)")
    print("\nNext: Run analysis notebooks in Jupyter")
    print("  - analysis/qtable_discrete_analysis.ipynb")
    print("  - analysis/qtable_continuous_analysis.ipynb")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
