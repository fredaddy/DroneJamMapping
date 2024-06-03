from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pybullet as p

# Create the environment
env = DroneJammingEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

# Train the model
total_timesteps = 100
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the model
model.save("ppo_drone")

# Load the model
model = PPO.load("ppo_drone")

print("done training")
# Evaluate the model
num_episodes = 10
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

# Plotting the results
# Plot rewards
plt.figure(figsize=(12, 5))
for idx, rewards in enumerate(all_rewards):
    plt.plot(rewards, label=f'Episode {idx + 1}')
plt.title('Rewards over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.show()

# Plot distances
plt.figure(figsize=(12, 5))
for idx, distances in enumerate(all_distances):
    plt.plot(distances, label=f'Episode {idx + 1}')
plt.title('Distance from Target over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.show()
