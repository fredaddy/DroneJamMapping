from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv
import numpy as np
import tqdm

# Create the environment
env = DroneJammingEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

# Train the model
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the model
model.save("ppo_drone")

# Load the model
model = PPO.load("ppo_drone")

print("done training")
# Evaluate the model
num_episodes = 1000
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        env.render()
    
    print(f"Episode {episode + 1}: Reward = {ep_reward}, Length = {ep_length}")

env.close()