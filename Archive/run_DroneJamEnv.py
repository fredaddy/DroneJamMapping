from stable_baselines3 import PPO
from DroneJamEnv import DroneJammingEnv

# Create the environment
env = DroneJammingEnv()

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

# Train the model
total_timesteps = 10
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("ppo_drone")

# Load the model
model = PPO.load("ppo_drone")

# Evaluate the model
num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        env.render()

    print(f"Episode {episode + 1}: Reward = {ep_reward}")

env.close()
