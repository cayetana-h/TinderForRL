import gymnasium as gym
import numpy as np
import yaml
import os
import sys
import importlib.util
from torch.utils.tensorboard import SummaryWriter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH  = os.path.abspath(os.path.join(CURRENT_DIR, "..", "agents", "agent_qtable.py"))

spec = importlib.util.spec_from_file_location("agent_qtable", AGENT_PATH)
agent_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_mod)
QTableAgent = agent_mod.QTableAgent
print("Agent loaded from:", AGENT_PATH)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def shape_reward(obs, obs_next, gamma):
    """
    Velocity-based potential shaping.

    phi(s) = |velocity|
    F(s,s') = gamma * phi(s') - phi(s)

    Rewards gaining speed regardless of direction.
    This directly incentivises the rocking strategy:
    the car must build speed left AND right to escape.
    Unlike position-based shaping, there is no local optimum
    where the car can sit still and collect reward.
    """
    return gamma * abs(obs_next[1]) - abs(obs[1])


def train():
    config_path = os.path.join(CURRENT_DIR, "..", "config", "qtable_discrete.yaml")
    config = load_config(config_path)

    env = gym.make("MountainCar-v0")

    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=config["num_bins"],
        num_actions=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
    )

    use_shaping   = config.get("use_reward_shaping", True)
    shaping_scale = config.get("shaping_scale", 300.0)

    raw_rewards    = []
    shaped_rewards = []
    successes      = 0

    # TensorBoard logging
    writer = SummaryWriter(log_dir="runs/qtable_discrete")

    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        state = agent.discretize_state(obs)
        total_raw    = 0
        total_shaped = 0

        for step in range(config["max_steps"]):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            raw_r = reward  # always track the real reward separately

            if use_shaping and not terminated:
                # Only shape non-terminal steps
                # Terminal step already has a clear signal (-1 at goal)
                bonus  = shape_reward(obs, obs_next, agent.gamma)
                reward = reward + shaping_scale * bonus

            next_state = agent.discretize_state(obs_next)
            agent.update(state, action, reward, next_state, done)

            obs   = obs_next
            state = next_state
            total_raw    += raw_r
            total_shaped += reward

            if done:
                if terminated:
                    successes += 1
                break

        agent.decay_epsilon()
        raw_rewards.append(total_raw)
        shaped_rewards.append(total_shaped)

        # Log to TensorBoard
        writer.add_scalar("Reward/raw", total_raw, episode)
        writer.add_scalar("Reward/shaped", total_shaped, episode)
        writer.add_scalar("Metrics/steps", step, episode)
        writer.add_scalar("Metrics/epsilon", agent.epsilon, episode)

        if episode % 500 == 0:
            recent_raw = raw_rewards[-100:] if episode >= 100 else raw_rewards
            print(
                f"Ep {episode:5d} | "
                f"Raw avg: {np.mean(recent_raw):7.1f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Successes so far: {successes}"
            )

        # Greedy evaluation every 2000 episodes
        if episode % 2000 == 0 and episode > 0:
            eval_env      = gym.make("MountainCar-v0")
            eval_wins     = 0
            for _ in range(20):
                o, _ = eval_env.reset()
                for _ in range(200):
                    s = agent.discretize_state(o)
                    a = int(np.argmax(agent.q_table[s]))
                    o, _, term, trunc, _ = eval_env.step(a)
                    if term:
                        eval_wins += 1
                        break
                    if trunc:
                        break
            print(f"  → Greedy eval: {eval_wins}/20")
            writer.add_scalar("Evaluation/greedy_wins", eval_wins, episode)

    writer.close()

    os.makedirs("../results/models",  exist_ok=True)
    os.makedirs("../results/metrics", exist_ok=True)

    agent.save("../results/models/q_table.npy")
    np.save("../results/metrics/rewards.npy",        np.array(raw_rewards))
    np.save("../results/metrics/rewards_shaped.npy", np.array(shaped_rewards))

    print(f"\nDone. Successes: {successes}/{config['num_episodes']}")
    print(f"Q-table: {agent.q_table.shape} | Max Q: {np.max(agent.q_table):.4f}")
    print(f"TensorBoard logs saved to: runs/qtable_discrete")


if __name__ == "__main__":
    train()