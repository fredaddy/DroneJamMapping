import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv

# Define the path for saving results
save_path = '/Users/fredaddy/Desktop/DroneJamMapping/results/'

# Load the trained model
model = PPO.load(os.path.join(save_path, "ppo_drone"))

# Create the environment
env = DroneJammingEnv()

# Evaluate the model
num_episodes = 100
all_rewards = []
all_distances = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    ep_rewards = []
    ep_distances = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward)
        current_position = obs[:3]
        distance = np.linalg.norm(current_position - env.jamming_source)
        ep_distances.append(distance)
        env.render()

    all_rewards.append(ep_rewards)
    all_distances.append(ep_distances)
    print(f"Episode {episode + 1}: Reward = {sum(ep_rewards)}, Length = {len(ep_rewards)}")

env.close()

# Plotting and saving plots
def plot_data(data, ylabel, title, file_name):
    plt.figure(figsize=(12, 5))
    plt.plot(np.max(data, axis=0), label='Max')
    plt.plot(np.min(data, axis=0), label='Min')
    plt.plot(np.median(data, axis=0), label='Median')
    plt.fill_between(range(len(data[0])), np.min(data, axis=0), np.max(data, axis=0), alpha=0.2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

plot_data(all_rewards, 'Reward', 'High-Med-Low Rewards over Iterations', 'rewards_over_iterations.png')
plot_data(all_distances, 'Distance', 'High-Med-Low Distance from Target over Iterations', 'distance_over_iterations.png')
