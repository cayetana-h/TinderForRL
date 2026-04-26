"""
Quick validation script to check if all dependencies are installed
and all required files/directories exist.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    """Verify all required packages are installed."""
    print("Checking dependencies...")
    required = [
        "numpy",
        "gymnasium",
        "yaml",
        "matplotlib",
        "sklearn",
        "stable_baselines3",
        "torch",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    return len(missing) == 0


def check_structure():
    """Verify project structure is complete."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "agents",
        "config",
        "training",
        "analysis",
        "results",
        "results/models",
        "results/metrics",
    ]
    
    missing = []
    for directory in required_dirs:
        path = os.path.join(CURRENT_DIR, directory)
        if os.path.isdir(path):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (MISSING)")
            missing.append(directory)
    
    return len(missing) == 0


def check_files():
    """Verify all required files exist."""
    print("\nChecking required files...")
    
    required_files = [
        "agents/agent_qtable.py",
        "config/qtable_discrete.yaml",
        "config/qtable_continuous.yaml",
        "config/deeprl.yaml",
        "training/train_qtable_discrete.py",
        "training/train_continuous_qtable.py",
        "training/train_continuous_deeprl.py",
        "analysis/compare_all_approaches.py",
        "analysis/visualize_policies.py",
        "run_all_training.py",
        "requirements.txt",
        "README.md",
    ]
    
    missing = []
    for filepath in required_files:
        full_path = os.path.join(CURRENT_DIR, filepath)
        if os.path.isfile(full_path):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} (MISSING)")
            missing.append(filepath)
    
    return len(missing) == 0


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("TINDER FOR RL - SETUP VALIDATION")
    print("=" * 70 + "\n")
    
    deps_ok = check_dependencies()
    struct_ok = check_structure()
    files_ok = check_files()
    
    print("\n" + "=" * 70)
    if deps_ok and struct_ok and files_ok:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext: python run_all_training.py")
    else:
        print("✗ SOME CHECKS FAILED")
        if not deps_ok:
            print("  → Run: pip install -r requirements.txt")
        if not struct_ok:
            print("  → Create missing directories manually")
        if not files_ok:
            print("  → Check project files exist")
    print("=" * 70 + "\n")
    
    return deps_ok and struct_ok and files_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
