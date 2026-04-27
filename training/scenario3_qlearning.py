import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class DiscreteActionCostWrapper(gym.Wrapper):
    def __init__(self, env, action_cost=0.001, goal_bonus=100):
        super().__init__(env)
        self.action_cost = action_cost
        self.goal_bonus = goal_bonus
        self.neutral_action = 1
        self.prev_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_pos = obs[0]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        pos, vel = obs
        progress = pos - self.prev_pos

        reward = -1.0
        reward += 100.0 * progress
        reward += 10.0 * abs(vel)

        if action != self.neutral_action:
            reward -= self.action_cost

        if terminated:
            reward += self.goal_bonus

        self.prev_pos = pos

        return obs, reward, terminated, truncated, info


env = DiscreteActionCostWrapper(gym.make("MountainCar-v0"))

episodes = 30000
max_steps = 200

bins = [30, 30]
alpha = 0.2
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9997

q_table = np.zeros((bins[0], bins[1], env.action_space.n))

state_low = env.observation_space.low
state_high = env.observation_space.high


def discretize(obs):
    ratios = (obs - state_low) / (state_high - state_low)
    indices = (ratios * np.array(bins)).astype(int)
    return tuple(np.clip(indices, 0, np.array(bins) - 1))


rewards = []
fuel_usage = []
successes = []

for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)

    total_reward = 0
    fuel = 0
    success = False

    for step in range(max_steps):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_obs)

        if action != 1:
            fuel += 1

        if terminated:
            target = reward
            success = True
        else:
            target = reward + gamma * np.max(q_table[next_state])

        q_table[state][action] += alpha * (target - q_table[state][action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards.append(total_reward)
    fuel_usage.append(fuel)
    successes.append(success)

    if ep % 1000 == 0:
        recent_success = np.mean(successes[-1000:]) if len(successes) >= 1000 else np.mean(successes)
        print(
            f"Episode {ep} | Reward: {total_reward:.2f} | "
            f"Fuel: {fuel} | Success: {success} | "
            f"Recent success: {recent_success:.2%} | Epsilon: {epsilon:.3f}"
        )


np.save("scenario3_qlearning_min_fuel_qtable.npy", q_table)
np.save("scenario3_qlearning_rewards.npy", np.array(rewards))
np.save("scenario3_qlearning_fuel.npy", np.array(fuel_usage))

print("\nFinal results:")
print(f"Success rate last 1000 episodes: {np.mean(successes[-1000:]):.2%}")
print(f"Average fuel last 1000 episodes: {np.mean(fuel_usage[-1000:]):.2f}")
print(f"Average reward last 1000 episodes: {np.mean(rewards[-1000:]):.2f}")

plt.plot(rewards)
plt.title("Scenario 3 Q-learning Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

plt.plot(fuel_usage)
plt.title("Scenario 3 Q-learning Fuel Usage")
plt.xlabel("Episode")
plt.ylabel("Non-neutral Actions")
plt.show()