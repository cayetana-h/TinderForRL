from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.io import ensure_dir, load_json


ROOT = Path(__file__).resolve().parent


def main():
    metrics_dir = ROOT / "results" / "metrics" / "qtable_action_cost"
    comparison_dir = ensure_dir(ROOT / "results" / "comparison")

    rewards_path = metrics_dir / "rewards.npy"
    shaped_path = metrics_dir / "shaped_rewards.npy"
    costs_path = metrics_dir / "costs.npy"
    summary_path = metrics_dir / "summary.json"

    if not rewards_path.exists():
        print("No qtable action-cost metrics found.")
        return

    rewards = np.load(rewards_path)
    shaped_rewards = np.load(shaped_path) if shaped_path.exists() else None
    costs = np.load(costs_path) if costs_path.exists() else None

    if summary_path.exists():
        print(load_json(summary_path))

    plt.figure(figsize=(11, 4))
    plt.plot(rewards, label="raw reward")
    if shaped_rewards is not None:
        plt.plot(shaped_rewards, label="shaped reward")
    plt.title("Q-table action-cost rewards")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_dir / "qtable_action_cost_rewards.png", dpi=160, bbox_inches="tight")
    plt.close()

    if costs is not None:
        plt.figure(figsize=(11, 4))
        plt.plot(costs, label="cost")
        plt.title("Q-table action-cost per episode")
        plt.xlabel("Episode")
        plt.ylabel("Cost")
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_dir / "qtable_action_cost_costs.png", dpi=160, bbox_inches="tight")
        plt.close()

    print("Continuous plots saved.")


if __name__ == "__main__":
    main()