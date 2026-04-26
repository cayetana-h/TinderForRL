"""
Post-training interpretability analysis of Q-table policies.

This script extracts the learned policy from a trained Q-table and fits a
DecisionTreeClassifier to explain which state features drive action selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os


def load_q_table(q_table_path):
    """Load saved Q-table."""
    return np.load(q_table_path)


def extract_policy_data(q_table, num_bins, env_bounds):
    """
    Extract (state_features, best_action) pairs from Q-table.

    Args:
        q_table: Q-table array of shape (bins_pos, bins_vel, num_actions)
        num_bins: List [bins_pos, bins_vel]
        env_bounds: Dict with 'low' and 'high' for position and velocity

    Returns:
        (state_features, actions) where each row is a discretized state
    """
    states = []
    actions = []

    pos_low, vel_low = env_bounds["low"]
    pos_high, vel_high = env_bounds["high"]
    bins_pos, bins_vel = num_bins

    # Iterate over all discretized states
    for i in range(bins_pos):
        for j in range(bins_vel):
            # Get best action for this state
            best_action = int(np.argmax(q_table[i, j]))

            # Convert bin indices to approximate state values
            pos_val = pos_low + (i + 0.5) * (pos_high - pos_low) / bins_pos
            vel_val = vel_low + (j + 0.5) * (vel_high - vel_low) / bins_vel

            states.append([pos_val, vel_val])
            actions.append(best_action)

    return np.array(states), np.array(actions)


def fit_policy_tree(states, actions, max_depth=4):
    """
    Fit a decision tree to the policy.

    Args:
        states: Array of state features (position, velocity)
        actions: Array of best actions
        max_depth: Maximum tree depth

    Returns:
        Fitted DecisionTreeClassifier
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(states, actions)
    return clf


def plot_tree_structure(clf, feature_names=None, class_names=None, filepath=None):
    """
    Plot and optionally save decision tree structure.

    Args:
        clf: Fitted DecisionTreeClassifier
        feature_names: Names of features
        class_names: Names of classes (actions)
        filepath: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(
        clf,
        feature_names=feature_names or ["position", "velocity"],
        class_names=class_names or ["left", "neutral", "right"],
        filled=True,
        ax=ax,
        fontsize=10,
    )

    plt.tight_layout()

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Tree plot saved to {filepath}")

    plt.show()

    return fig, ax


def plot_feature_importance(clf, feature_names=None, filepath=None):
    """
    Plot feature importances from the decision tree.

    Args:
        clf: Fitted DecisionTreeClassifier
        feature_names: Names of features
        filepath: If provided, save figure to this path
    """
    importances = clf.feature_importances_
    feature_names = feature_names or ["position", "velocity"]

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.argsort(importances)[::-1]

    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices])
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance in Learned Policy")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to {filepath}")

    plt.show()

    return fig, ax


def print_policy_summary(clf, feature_names=None, class_names=None):
    """
    Print human-readable summary of the learned policy.

    Args:
        clf: Fitted DecisionTreeClassifier
        feature_names: Names of features
        class_names: Names of classes
    """
    feature_names = feature_names or ["position", "velocity"]
    class_names = class_names or ["left", "neutral", "right"]

    print("\n" + "=" * 60)
    print("POLICY INTERPRETABILITY SUMMARY")
    print("=" * 60)

    importances = clf.feature_importances_
    print(f"\nFeature Importances:")
    for fname, imp in zip(feature_names, importances):
        print(f"  {fname:12s}: {imp:.4f}")

    print(f"\nTree Depth: {clf.get_depth()}")
    print(f"Num Leaves: {clf.get_n_leaves()}")
    print(f"Num Nodes: {clf.tree_.node_count}")

    # Estimate policy complexity
    complexity_score = clf.tree_.node_count / 100.0
    print(f"Policy Complexity Score: {complexity_score:.2f}")

    print("\nTop Decision Rules:")
    print("(First few splits in the tree)")
    _print_tree_rules(clf, feature_names, class_names, depth=0, max_depth=2)

    print("\n" + "=" * 60)


def _print_tree_rules(
    clf, feature_names, class_names, node=0, depth=0, max_depth=3, prefix=""
):
    """Recursively print tree decision rules."""
    if depth > max_depth:
        return

    tree = clf.tree_

    if tree.feature[node] != -2:  # Not a leaf
        name = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        print(f"{prefix}if {name} <= {threshold:.3f}:")
        _print_tree_rules(
            clf,
            feature_names,
            class_names,
            tree.children_left[node],
            depth + 1,
            max_depth,
            prefix + "  ",
        )
        print(f"{prefix}else ({name} > {threshold:.3f}):")
        _print_tree_rules(
            clf,
            feature_names,
            class_names,
            tree.children_right[node],
            depth + 1,
            max_depth,
            prefix + "  ",
        )
    else:  # Leaf
        class_counts = tree.value[node][0]
        majority_class = np.argmax(class_counts)
        print(f"{prefix}→ predict {class_names[majority_class]}")


def analyze_qtable_policy(
    q_table_path,
    num_bins,
    env_bounds,
    output_dir="results/interpretability",
    max_tree_depth=4,
):
    """
    Complete interpretability analysis pipeline.

    Args:
        q_table_path: Path to saved Q-table
        num_bins: [bins_pos, bins_vel]
        env_bounds: Dict with 'low' and 'high' keys
        output_dir: Directory for saving plots
        max_tree_depth: Maximum depth of decision tree
    """
    print(f"Loading Q-table from {q_table_path}...")
    q_table = load_q_table(q_table_path)

    print("Extracting policy from Q-table...")
    states, actions = extract_policy_data(q_table, num_bins, env_bounds)

    print(f"Fitting decision tree (max_depth={max_tree_depth})...")
    clf = fit_policy_tree(states, actions, max_depth=max_tree_depth)

    # Print summary
    print_policy_summary(clf)

    # Plot tree
    tree_filepath = os.path.join(output_dir, "policy_tree.png")
    plot_tree_structure(
        clf,
        feature_names=["position", "velocity"],
        class_names=["left", "neutral", "right"],
        filepath=tree_filepath,
    )

    # Plot feature importances
    importance_filepath = os.path.join(output_dir, "feature_importances.png")
    plot_feature_importance(
        clf, feature_names=["position", "velocity"], filepath=importance_filepath
    )

    print(f"\nOutputs saved to {output_dir}/")
    return clf, states, actions


if __name__ == "__main__":
    import gymnasium as gym

    # Configuration
    q_table_path = "results/models/q_table.npy"
    env = gym.make("MountainCar-v0")
    num_bins = [200, 200]  # Match training config
    env_bounds = {"low": env.observation_space.low, "high": env.observation_space.high}

    # Run analysis
    clf, states, actions = analyze_qtable_policy(
        q_table_path,
        num_bins,
        env_bounds,
        output_dir="results/interpretability",
        max_tree_depth=4,
    )
