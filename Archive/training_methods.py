import numpy as np
import gym
import random
from DroneJamEnv_3 import DroneJammingEnv
# Assuming 'DroneJammingEnv' is already defined and imported

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * target
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def discretize_signal_strength(signal_strength):
    # Assume signal strength ranges from 0 to a maximum observed value and discretize it
    return int(signal_strength * 100)  # Example discretization

def train_agent(episodes):
    env = DroneJammingEnv()
    state_size = 10000  # Example state size
    action_size = env.action_space.n

    agent = QLearningAgent(state_size, action_size)

    for e in range(episodes):
        state = discretize_signal_strength(env.reset()[0])
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_signal_strength(next_state[0])
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        print(f"Episode: {e+1}/{episodes}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")

    env.close()

# Run training
train_agent(1000)
