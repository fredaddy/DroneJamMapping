import numpy as np
from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv
import matplotlib.pyplot as plt
import os
import pybullet as p

# Define the path for saving results
save_path = '/Users/fredaddy/Desktop/DroneJamMapping/results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create the environment
env = DroneJammingEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

# Train the model
total_timesteps = 100000
try:
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
except Exception as e:
    print(f"Error during training: {e}")

# Save the model
model.save("ppo_drone")

# Load the model
model = PPO.load("ppo_drone")

print("done training")

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
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_rewards.append(reward)
            current_position = obs[:3]
            distance = np.linalg.norm(current_position - env.jamming_source)
            ep_distances.append(distance)
            env.render()
        except Exception as e:
            print(f"Error during episode {episode + 1}: {e}")
            done = True  # End the episode if an error occurs

    all_rewards.append(ep_rewards)
    all_distances.append(ep_distances)
    print(f"Episode {episode + 1}: Reward = {sum(ep_rewards)}, Length = {len(ep_rewards)}")

env.close()

# Aggregate data for plotting
max_rewards = np.zeros(max(map(len, all_rewards)))
min_rewards = np.zeros(max(map(len, all_rewards)))
med_rewards = np.zeros(max(map(len, all_rewards)))
max_distances = np.zeros(max(map(len, all_distances)))
min_distances = np.zeros(max(map(len, all_distances)))
med_distances = np.zeros(max(map(len, all_distances)))

for t in range(len(max_rewards)):
    iteration_rewards = [rewards[t] for rewards in all_rewards if len(rewards) > t]
    iteration_distances = [distances[t] for distances in all_distances if len(distances) > t]

    max_rewards[t] = np.max(iteration_rewards)
    min_rewards[t] = np.min(iteration_rewards)
    med_rewards[t] = np.median(iteration_rewards)
    max_distances[t] = np.max(iteration_distances)
    min_distances[t] = np.min(iteration_distances)
    med_distances[t] = np.median(iteration_distances)

# Plotting the results
# Plot rewards
plt.figure(figsize=(12, 5))
plt.plot(max_rewards, label='Max Reward')
plt.plot(min_rewards, label='Min Reward')
plt.plot(med_rewards, label='Median Reward')
plt.fill_between(range(len(max_rewards)), min_rewards, max_rewards, alpha=0.2)
plt.title('High-Med-Low Rewards over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.savefig(os.path.join(save_path, 'high_med_low_rewards_over_iterations.png'))
plt.close()

# Plot distances
plt.figure(figsize=(12, 5))
plt.plot(max_distances, label='Max Distance')
plt.plot(min_distances, label='Min Distance')
plt.plot(med_distances, label='Median Distance')
plt.fill_between(range(len(max_distances)), min_distances, max_distances, alpha=0.2)
plt.title('High-Med-Low Distance from Target over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.savefig(os.path.join(save_path, 'high_med_low_distance_from_target_over_iterations.png'))
plt.close()
