"""
Train DQN agent for intelligent elevator scheduling.

Demonstrates deep reinforcement learning applied to real-world problem.
Uses experience replay and target networks for stable training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from elevator_env import ElevatorEnv


class DQNNetwork(nn.Module):
    """Deep Q-Network with 2 hidden layers."""
    
    def __init__(self, state_size=5, action_size=3, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Learning agent with experience replay."""
    
    def __init__(self, state_size=5, action_size=3, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Networks
        self.device = torch.device("cpu")
        self.network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.update_target_freq = 100  # Update target network every N steps
        self.train_step = 0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return torch.argmax(q_values[0]).item()
    
    def replay(self, batch_size):
        """Experience replay with mini-batch training."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
    
    def save(self, path):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)
    
    def load(self, path):
        """Load model weights."""
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.network.state_dict())


def train_dqn(num_episodes=1000, env_name="Elevator-DQN"):
    """Train DQN agent on elevator environment."""
    
    env = ElevatorEnv(num_floors=5, max_steps=300, passenger_arrival_prob=0.3)
    agent = DQNAgent(state_size=5, action_size=3)
    
    # Tracking
    episode_rewards = []
    episode_energy = []
    episode_deliveries = []
    
    print(f"Starting DQN training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay(agent.batch_size)
            
            episode_reward += reward
            state = next_state
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_energy.append(env.energy_consumed)
        episode_deliveries.append(env.passengers_delivered)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_energy = np.mean(episode_energy[-100:])
            avg_delivered = np.mean(episode_deliveries[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:7.2f} | "
                  f"Energy: {avg_energy:6.1f} | Delivered: {avg_delivered:5.1f}")
    
    # Save results
    agent.save("dqn_elevator_model.pt")
    np.save("dqn_rewards.npy", episode_rewards)
    np.save("dqn_energy.npy", episode_energy)
    np.save("dqn_deliveries.npy", episode_deliveries)
    
    print("\nTraining complete! Results saved.")
    
    return episode_rewards, episode_energy, episode_deliveries


if __name__ == "__main__":
    rewards, energy, deliveries = train_dqn(num_episodes=500)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Smoothed rewards
    window = 50
    smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smooth_rewards)
    axes[0].set_title("DQN: Episode Reward (smoothed)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    
    # Energy consumption
    smooth_energy = np.convolve(energy, np.ones(window)/window, mode='valid')
    axes[1].plot(smooth_energy, color='orange')
    axes[1].set_title("DQN: Energy Consumption (smoothed)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Energy Used")
    axes[1].grid(True, alpha=0.3)
    
    # Passengers delivered
    smooth_delivered = np.convolve(deliveries, np.ones(window)/window, mode='valid')
    axes[2].plot(smooth_delivered, color='green')
    axes[2].set_title("DQN: Passengers Delivered (smoothed)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("dqn_training_results.png", dpi=150)
    print("Plot saved as dqn_training_results.png")
