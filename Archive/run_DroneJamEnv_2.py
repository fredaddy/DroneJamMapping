import cv2
import numpy as np
from DroneJamEnv import DroneJammingEnv
from collections import deque

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Fast decay for quick learning

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * target
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def discretize_signal_strength(signal_strength):
    return min(int(signal_strength * 100), 9999)  # Ensuring index stays within bounds

def train_agent(episodes=10):  # Reduced number of episodes for quick test
    env = DroneJammingEnv()
    state_size = 10000
    action_size = env.action_space.n
    agent = QLearningAgent(state_size, action_size)
    frames = []

    for e in range(episodes):
        state = discretize_signal_strength(env.reset()[0])
        total_reward = 0
        done = False
        step = 0
        while not done and step < 50:  # Limit steps to prevent long episodes
            action = agent.choose_action(state)
            next_state, reward, done, img = env.step(action)
            frames.append(img)
            next_state = discretize_signal_strength(next_state[0])
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

        print(f"Episode: {e+1}/{episodes}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")

    env.close()
    return frames

def save_video(frames, filename='drone_simulation.mp4'):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

frames = train_agent()
save_video(frames)
