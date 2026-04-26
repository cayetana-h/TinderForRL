"""
Train Q-Learning agent for intelligent elevator scheduling (baseline).

Simple tabular Q-learning approach for comparison with DQN.
Demonstrates state discretization and off-policy learning.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from elevator_env import ElevatorEnv


class QLearningAgent:
    """Tabular Q-Learning agent for discrete state spaces."""
    
    def __init__(self, num_floors=5, action_size=3, learning_rate=0.1, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.num_floors = num_floors
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table: (current_floor, destination_floor, requests_discretized, direction) -> action_values
        # We discretize the requests and energy for tabular representation
        self.q_table = {}
        self.state_count = 0
    
    def _discretize_state(self, observation):
        """Convert continuous observation to discrete state key."""
        current_floor = int(observation[0])
        dest_floor = int(observation[1])
        requests_encoded = int(observation[2])
        direction = int(observation[3]) + 1  # Convert -1,0,1 to 0,1,2
        energy_discretized = int(observation[4] * 10)  # 0-10 levels
        
        state = (current_floor, dest_floor, requests_encoded, direction, energy_discretized)
        return state
    
    def get_q_values(self, state):
        """Get Q-values for a state, initializing if necessary."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
            self.state_count += 1
        return self.q_table[state]
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def update_q_value(self, state, action, reward, next_state):
        """Q-learning update rule."""
        current_q = self.get_q_values(state)[action]
        next_q_max = np.max(self.get_q_values(next_state))
        
        # Q(s,a) <- Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_q_max - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Reduce exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        """Save Q-table."""
        np.save(path, self.q_table)
    
    def load(self, path):
        """Load Q-table."""
        self.q_table = np.load(path, allow_pickle=True).item()


def train_qlearning(num_episodes=1000):
    """Train Q-Learning agent on elevator environment."""
    
    env = ElevatorEnv(num_floors=5, max_steps=300, passenger_arrival_prob=0.3)
    agent = QLearningAgent(num_floors=5, action_size=3)
    
    # Tracking
    episode_rewards = []
    episode_energy = []
    episode_deliveries = []
    unique_states = []
    
    print(f"Starting Q-Learning training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent._discretize_state(obs)
        episode_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            obs, reward, done, _, info = env.step(action)
            next_state = agent._discretize_state(obs)
            
            # Q-learning update
            agent.update_q_value(state, action, reward, next_state)
            
            episode_reward += reward
            state = next_state
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_energy.append(env.energy_consumed)
        episode_deliveries.append(env.passengers_delivered)
        unique_states.append(agent.state_count)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_energy = np.mean(episode_energy[-100:])
            avg_delivered = np.mean(episode_deliveries[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Energy: {avg_energy:6.1f} | Delivered: {avg_delivered:5.1f} | "
                  f"States: {agent.state_count}")
    
    # Save results
    agent.save("qlearning_elevator_qtable.npy")
    np.save("qlearning_rewards.npy", episode_rewards)
    np.save("qlearning_energy.npy", episode_energy)
    np.save("qlearning_deliveries.npy", episode_deliveries)
    
    print("\nTraining complete! Results saved.")
    print(f"Total unique states discovered: {agent.state_count}")
    
    return episode_rewards, episode_energy, episode_deliveries


if __name__ == "__main__":
    rewards, energy, deliveries = train_qlearning(num_episodes=500)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Smoothed rewards
    window = 50
    smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smooth_rewards)
    axes[0].set_title("Q-Learning: Episode Reward (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    
    # Energy consumption
    smooth_energy = np.convolve(energy, np.ones(window)/window, mode='valid')
    axes[1].plot(smooth_energy, color='orange')
    axes[1].set_title("Q-Learning: Energy Consumption (smoothed)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Energy Used")
    axes[1].grid(True, alpha=0.3)
    
    # Passengers delivered
    smooth_delivered = np.convolve(deliveries, np.ones(window)/window, mode='valid')
    axes[2].plot(smooth_delivered, color='green')
    axes[2].set_title("Q-Learning: Passengers Delivered (smoothed)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("qlearning_training_results.png", dpi=150)
    print("Plot saved as qlearning_training_results.png")
