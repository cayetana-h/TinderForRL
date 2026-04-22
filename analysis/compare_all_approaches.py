from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.io import ensure_dir, load_json, save_csv_rows

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    ("Q-table discrete", ROOT / "results" / "metrics" / "qtable_discrete" / "summary.json"),
    ("Q-table action cost", ROOT / "results" / "metrics" / "qtable_action_cost" / "summary.json"),
    ("TD3 continuous", ROOT / "results" / "metrics" / "td3_continuous" / "summary.json"),
    ("SAC continuous", ROOT / "results" / "metrics" / "sac_continuous" / "summary.json"),
]


def load_summaries():
    rows = []
    for method_name, summary_path in EXPERIMENTS:
        if not summary_path.exists():
            continue
        summary = dict(load_json(summary_path))
        summary["method"] = method_name
        rows.append(summary)
    return rows


def plot_bar(rows, key, title, ylabel, output_path):
    if not rows:
        return

    labels = [row["method"] for row in rows]
    values = [float(row.get(key, 0.0)) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    rows = load_summaries()
    if not rows:
        print("No summary.json files found under results/metrics/")
        return

    comparison_dir = ensure_dir(ROOT / "results" / "comparison")
    save_csv_rows(rows, comparison_dir / "comparison_summary.csv")

    plot_bar(rows, "success_rate", "Success rate by approach", "Success rate", comparison_dir / "success_rate.png")
    plot_bar(rows, "mean_reward", "Mean reward by approach", "Mean reward", comparison_dir / "mean_reward.png")
    plot_bar(rows, "mean_steps", "Mean steps by approach", "Mean steps", comparison_dir / "mean_steps.png")
    plot_bar(rows, "mean_cost", "Mean cost by approach", "Mean cost", comparison_dir / "mean_cost.png")

    best_success = max(rows, key=lambda row: float(row.get("success_rate", 0.0)))
    best_reward = max(rows, key=lambda row: float(row.get("mean_reward", -1e9)))

    notes = [
        f"Best success rate: {best_success['method']} ({float(best_success.get('success_rate', 0.0)):.3f})",
        f"Best mean reward: {best_reward['method']} ({float(best_reward.get('mean_reward', 0.0)):.3f})",
    ]
    (comparison_dir / "comparison_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    print("Comparison saved to", comparison_dir)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()