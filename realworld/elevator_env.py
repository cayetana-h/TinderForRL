"""
Intelligent Elevator Scheduling Environment (Gymnasium-compatible)

A realistic multi-floor elevator control task where the agent learns
to minimize passenger wait times and energy consumption.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import random


class ElevatorEnv(gym.Env):
    """
    Elevator scheduling environment.
    
    State: (current_floor, destination_floor, pending_requests_encoded, direction, energy_pct)
    Action: 0=up, 1=down, 2=stop
    Reward: -2*(avg_wait) - 0.5*(energy) + 5*(deliveries) - 0.1*(idle_moves)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, num_floors=5, max_steps=500, passenger_arrival_prob=0.3):
        """
        Initialize elevator environment.
        
        Args:
            num_floors: Number of floors (0 to num_floors-1)
            max_steps: Maximum steps per episode
            passenger_arrival_prob: Probability of new passenger each step
        """
        super().__init__()
        
        self.num_floors = num_floors
        self.max_steps = max_steps
        self.passenger_arrival_prob = passenger_arrival_prob
        
        # Observation: (current_floor, dest_floor, requests_bitmask, direction, energy_pct)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0], dtype=np.float32),
            high=np.array([num_floors-1, num_floors-1, 2**num_floors-1, 1, 100], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: 0=up, 1=down, 2=stop
        self.action_space = spaces.Discrete(3)
        
        # State variables
        self.current_floor = 0
        self.destination_floor = 0
        self.direction = 0  # -1=down, 0=stopped, 1=up
        self.pending_requests = set()  # Floors with waiting passengers
        self.passengers_in_elevator = {}  # {floor: count}
        
        # Metrics
        self.energy_consumed = 0.0
        self.total_wait_time = 0.0
        self.passengers_delivered = 0
        self.step_count = 0
        self.idle_moves = 0
        
        # Passenger tracking
        self.passenger_queue = {}  # {floor: [dest_floor1, dest_floor2, ...]}
        for f in range(num_floors):
            self.passenger_queue[f] = []
    
    def _get_observation(self):
        """Convert state to observation tuple."""
        requests_encoded = 0
        for floor in self.pending_requests:
            requests_encoded |= (1 << floor)
        
        return np.array([
            float(self.current_floor),
            float(self.destination_floor),
            float(requests_encoded),
            float(self.direction),
            min(self.energy_consumed / 100.0, 1.0)  # Normalize energy 0-1
        ], dtype=np.float32)
    
    def _add_passengers(self):
        """Stochastically add new passengers at random floors."""
        for floor in range(self.num_floors):
            if random.random() < self.passenger_arrival_prob:
                dest = random.choice([f for f in range(self.num_floors) if f != floor])
                self.passenger_queue[floor].append(dest)
                self.pending_requests.add(floor)
    
    def _pick_up_passengers(self):
        """Pick up waiting passengers at current floor."""
        if self.current_floor in self.passenger_queue:
            passengers_here = self.passenger_queue[self.current_floor]
            for dest in passengers_here:
                if dest not in self.passengers_in_elevator:
                    self.passengers_in_elevator[dest] = 0
                self.passengers_in_elevator[dest] += 1
            
            self.passenger_queue[self.current_floor] = []
            if self.current_floor in self.pending_requests:
                self.pending_requests.discard(self.current_floor)
    
    def _drop_off_passengers(self):
        """Drop off passengers at their destination."""
        if self.current_floor in self.passengers_in_elevator:
            count = self.passengers_in_elevator[self.current_floor]
            self.passengers_delivered += count
            del self.passengers_in_elevator[self.current_floor]
    
    def _calculate_reward(self, action_taken):
        """Calculate reward based on action and state."""
        reward = 0.0
        
        # Penalize average wait time (accumulate across episode)
        self.total_wait_time += len(self.pending_requests)
        reward -= 0.5 * len(self.pending_requests)
        
        # Penalize energy consumption
        if action_taken in [0, 1]:  # Moving up or down
            reward -= 1.0
            self.energy_consumed += 1.0
        else:  # Stopping
            reward -= 0.1
            self.energy_consumed += 0.1
        
        # Reward passenger deliveries
        reward += 2.0 * self.passengers_delivered
        
        # Penalize idle moves (moving when no requests)
        if action_taken in [0, 1] and len(self.pending_requests) == 0:
            self.idle_moves += 1
            reward -= 0.5
        
        return reward
    
    def step(self, action):
        """Execute one step of environment."""
        self.step_count += 1
        
        # Add new passengers
        self._add_passengers()
        
        # Execute action
        if action == 0:  # Up
            if self.current_floor < self.num_floors - 1:
                self.current_floor += 1
                self.direction = 1
        elif action == 1:  # Down
            if self.current_floor > 0:
                self.current_floor -= 1
                self.direction = -1
        else:  # Stop
            self.direction = 0
            self._pick_up_passengers()
            self._drop_off_passengers()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check terminal condition
        terminated = self.step_count >= self.max_steps
        
        # Get observation
        obs = self._get_observation()
        
        # Info
        info = {
            "passengers_delivered": self.passengers_delivered,
            "energy_consumed": self.energy_consumed,
            "pending_requests": len(self.pending_requests)
        }
        
        return obs, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_floor = 0
        self.destination_floor = 0
        self.direction = 0
        self.pending_requests = set()
        self.passengers_in_elevator = {}
        
        self.energy_consumed = 0.0
        self.total_wait_time = 0.0
        self.passengers_delivered = 0
        self.step_count = 0
        self.idle_moves = 0
        
        for f in range(self.num_floors):
            self.passenger_queue[f] = []
        
        return self._get_observation(), {}
    
    def render(self):
        """Simple text rendering."""
        print(f"Step {self.step_count} | Floor: {self.current_floor} | "
              f"Pending: {len(self.pending_requests)} | "
              f"Energy: {self.energy_consumed:.1f} | "
              f"Delivered: {self.passengers_delivered}")


if __name__ == "__main__":
    # Test environment
    env = ElevatorEnv(num_floors=5, max_steps=100)
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation: {obs}")
    
    # Run random policy
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated:
            print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
