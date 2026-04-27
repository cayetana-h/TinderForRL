"""
Scenario 1: Discrete MountainCar, Minimum Steps (Standard)

Environment: MountainCar-v0
Reward: -1 per step (standard)
Goal: Reach the goal as quickly as possible

Algorithm: Q-learning (tabular)
"""

from __future__ import annotations
from pathlib import Path
import gymnasium as gym
import numpy as np
import json


# ============================================================================
# Q-TABLE AGENT CLASS
# ============================================================================

class QTableAgent:
    """Tabular Q-learning agent"""
    
    def __init__(
        self,
        state_low,
        state_high,
        num_bins,
        num_actions,
        learning_rate=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9998,
    ):
        self.num_bins = np.array(num_bins, dtype=int)
        self.num_actions = int(num_actions)
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        self.state_low = np.array(state_low, dtype=float)
        self.state_high = np.array(state_high, dtype=float)

        self.q_table = np.zeros(tuple(self.num_bins) + (self.num_actions,), dtype=np.float32)
        self.bin_width = np.maximum((self.state_high - self.state_low) / self.num_bins, 1e-12)

    def discretize_state(self, state):
        state = np.asarray(state, dtype=float)
        indices = (state - self.state_low) / self.bin_width
        indices = np.clip(indices.astype(int), 0, self.num_bins - 1)
        return tuple(indices)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state):
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state])
        target = float(reward) + (0.0 if done else self.gamma * best_next)
        idx = state + (int(action),)
        self.q_table[idx] += self.lr * (target - self.q_table[idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rolling_mean(data, window):
    """Calculate rolling mean"""
    if len(data) < window:
        return np.array([])
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ============================================================================
# TRAINING
# ============================================================================

def train():
    # Configuration
    SEED = 42
    NUM_BINS = [20, 20]
    NUM_EPISODES = 10000
    MAX_STEPS = 200
    LEARNING_RATE = 0.2
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9998
    
    np.random.seed(SEED)
    
    # Create environment
    env = gym.make("MountainCar-v0")
    
    # Create agent
    agent = QTableAgent(
        state_low=env.observation_space.low,
        state_high=env.observation_space.high,
        num_bins=NUM_BINS,
        num_actions=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    )
    
    # Training metrics
    rewards = []
    steps_list = []
    successes = []
    
    print("=" * 60)
    print("SCENARIO 1: Discrete MountainCar - Minimum Steps (Q-learning)")
    print("=" * 60)
    
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset(seed=SEED + episode)
        state = agent.discretize_state(obs)
        
        total_reward = 0.0
        terminated = False
        
        for step_count in range(1, MAX_STEPS + 1):
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = agent.discretize_state(obs_next)
            agent.update(state, action, reward, next_state, done)
            
            obs = obs_next
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        steps_list.append(step_count)
        successes.append(bool(terminated))
        
        agent.decay_epsilon()
        
        if episode % 500 == 0:
            recent = rewards[-100:] if len(rewards) >= 100 else rewards
            print(
                f"Ep {episode:5d} | "
                f"Avg reward (last 100): {np.mean(recent):8.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Successes: {sum(successes)}"
            )
    
    # Save results
    results_dir = ensure_dir("results/models")
    metrics_dir = ensure_dir("results/metrics/scenario1_qlearning")
    
    agent.save(results_dir / "scenario1_qlearning.npy")
    np.save(metrics_dir / "rewards.npy", np.array(rewards, dtype=np.float32))
    np.save(metrics_dir / "steps.npy", np.array(steps_list, dtype=np.int32))
    
    # Save summary
    summary = {
        "algorithm": "Q-learning",
        "scenario": "Scenario 1 - Discrete, Min Steps",
        "num_episodes": NUM_EPISODES,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": float(sum(successes) / len(successes)),
        "final_epsilon": float(agent.epsilon),
    }
    
    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Mean reward: {summary['mean_reward']:.2f}")
    print(f"Mean steps: {summary['mean_steps']:.2f}")
    print(f"Results saved to {metrics_dir}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    train()
