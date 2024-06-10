import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv
import pybullet as p

# Define the path for saving results
save_path = '/Users/fredaddy/Desktop/DroneJamMapping/results/'
folder_name = 'new_penalty_moving_away'

# Load the trained model
model = PPO.load(os.path.join(save_path, folder_name, "ppo_drone"))

# Create the environment
env = DroneJammingEnv()

# Evaluate the model
num_episodes = 20
max_steps_per_episode = 10000  # Maximum number of steps per episode
all_rewards = []
all_distances = []

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    ep_rewards = []
    ep_distances = []
    steps = 0  # Step counter for the current episode
    while not done and steps < max_steps_per_episode:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward)
        current_position = obs[:3]
        distance = np.linalg.norm(current_position - env.jamming_source)
        ep_distances.append(distance)
        
        try:
            env.render()
        except p.error as e:
            print(f"Render error: {e}")
        
        steps += 1  # Increment the step counter

    all_rewards.append(sum(ep_rewards))
    all_distances.append(np.mean(ep_distances))
    print(f"Episode {episode + 1}: Reward = {sum(ep_rewards)}, Length = {len(ep_rewards)}")

env.close()

# Plotting and saving plots
def plot_data(data, ylabel, title, file_name):
    plt.figure(figsize=(12, 5))
    plt.plot(data, label='Episode data')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(save_path, folder_name, file_name))
    plt.close()

plot_data(all_rewards, 'Total Reward', 'Total Rewards by Episode', 'total_rewards_by_episode.png')
plot_data(all_distances, 'Average Distance', 'Average Distance from Target by Episode', 'average_distance_by_episode.png')
