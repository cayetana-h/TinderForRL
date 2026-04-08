import numpy as np


class QTableAgent:
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
        epsilon_decay=0.9995,
    ):
        self.num_bins = np.array(num_bins, dtype=int)
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.state_low = np.array(state_low, dtype=float)
        self.state_high = np.array(state_high, dtype=float)

        # Optimistic initialization: -200 is the worst possible return
        # This encourages exploration of unvisited states
        self.q_table = np.ones(tuple(self.num_bins) + (num_actions,)) * (-200.0)

        # FIXED: divide by num_bins, not (num_bins - 1)
        self.bin_width = (self.state_high - self.state_low) / self.num_bins

    def discretize_state(self, state):
        """Convert continuous state to discrete bin indices."""
        indices = (state - self.state_low) / self.bin_width
        indices = np.clip(indices.astype(int), 0, self.num_bins - 1)
        return tuple(indices)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """Standard Q-learning update."""
        best_next = np.max(self.q_table[next_state])
        target = reward + (0.0 if done else self.gamma * best_next)
        error = target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.lr * error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)