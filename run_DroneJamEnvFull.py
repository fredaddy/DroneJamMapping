import os
import sys
from stable_baselines3 import PPO
from DroneJamEnvFull import DroneJammingEnv

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python run_DroneJamEnvFull.py <run_name>")
    sys.exit(1)

# Get the run_name from the command line arguments
run_name = sys.argv[1]

# Define the path for saving results
save_path = '/Users/fredaddy/Desktop/DroneJamMapping/results/'
run_path = os.path.join(save_path, run_name)

if not os.path.exists(run_path):
    os.makedirs(run_path)

# Create the environment
env = DroneJammingEnv()

# Initialize the PPO agent with adjusted parameters
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, learning_rate=3e-4, batch_size=64, clip_range=0.2)

# Train the model
total_timesteps = 1000000
try:
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    # Save the model
    model.save(os.path.join(run_path, "ppo_drone"))
    print(f"Model trained and saved successfully in {run_path}.")
except Exception as e:
    print(f"Error during training: {e}")

env.close()
