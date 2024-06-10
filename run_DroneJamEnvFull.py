import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
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

# Create the environment and wrap it with the Monitor
env = DroneJammingEnv()
env = Monitor(env, filename=os.path.join(run_path, "monitor.csv"))
env = DummyVecEnv([lambda: env])

# Initialize the PPO agent with adjusted parameters
model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, learning_rate=3e-3, batch_size=64, clip_range=0.2)

# Set up the evaluation environment and callback
eval_env = DummyVecEnv([lambda: Monitor(DroneJammingEnv(), filename=os.path.join(run_path, "eval_monitor.csv"))])
eval_callback = EvalCallback(eval_env, best_model_save_path=run_path, log_path=run_path, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False)

# Train the model
total_timesteps = 1e6
try:
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    # Save the model
    model.save(os.path.join(run_path, "ppo_drone"))
    print(f"Model trained and saved successfully in {run_path}.")
except Exception as e:
    print(f"Error during training: {e}")

env.close()
