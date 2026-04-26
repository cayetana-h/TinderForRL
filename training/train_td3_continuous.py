from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from utils.io import ensure_dir, load_yaml, save_json
from utils.wrappers import ContinuousExtraActionCostWrapper


ROOT = Path(__file__).resolve().parents[1]


def make_env(seed: int, extra_cost: float = 0.0):
    env = gym.make("MountainCarContinuous-v0")
    env = ContinuousExtraActionCostWrapper(env, extra_cost_coefficient=extra_cost)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train(config_path: str | Path = ROOT / "config" / "deeprl.yaml"):
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    model_dir = ensure_dir(ROOT / "results" / "models" / "td3_continuous")
    metrics_dir = ensure_dir(ROOT / "results" / "metrics" / "td3_continuous")

    env = make_env(seed=seed, extra_cost=float(cfg.get("extra_action_cost", 0.0)))

    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape[0]),
        sigma=float(cfg.get("td3_noise_scale", 0.1)) * np.ones(env.action_space.shape[0]),
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.005)),
        action_noise=action_noise,
        buffer_size=int(cfg.get("buffer_size", 100000)),
        batch_size=int(cfg.get("batch_size", 64)),
        learning_starts=int(cfg.get("learning_starts", 1000)),
        policy_delay=int(cfg.get("policy_delay", 2)),
        learning_rate=float(cfg.get("td3_learning_rate", 0.001)),
        verbose=1,
        seed=seed,
        tensorboard_log=str(ROOT / "results" / "tensorboard" / "td3"),
    )

    total_timesteps = int(cfg.get("total_timesteps", 100000))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model.save(str(model_dir / "td3_continuous"))
    save_json(
        {
            "algorithm": "TD3",
            "total_timesteps": total_timesteps,
            "seed": seed,
            "config": cfg,
        },
        metrics_dir / "train_metadata.json",
    )

    env.close()
    print(f"TD3 saved to {model_dir / 'td3_continuous.zip'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "deeprl.yaml"))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()