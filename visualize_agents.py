import numpy as np
from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv
import time
import pybullet as p

# Create the environment with GUI mode
env = DroneJammingEnv(action_scale=5, connection_mode=p.GUI)

# Load the trained model
model = PPO.load("/Users/fredaddy/Desktop/DroneJamMapping/results/learning_rate_3e3_n_steps_1024/ppo_drone")

# Run the simulation
done = False
while not done:
    obs = env.reset()  # Reset environment which should place a new jamming source randomly

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        #time.sleep(1.0 / 60.0)  # Adjust as necessary for your simulation speed

        # Optional: Output the current position and signal strength for debugging
        current_position, current_signal = obs[:3], obs[-1]
        print(f"Drone Position: {current_position}, Signal Strength: {current_signal}")

# Close the environment
env.close()
