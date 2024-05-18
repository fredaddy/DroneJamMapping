# DroneJamMapping
Reinforcement Learning module that trains a drone (agent) to identify point-source jammer autonomously and return coordinates to its operator.

## Installation

First, install the required dependencies:

```bash
pip install pybullet gym stable-baselines3 opencv-python
```
## Training the model

```python
from stable_baselines3 import PPO
from DroneJammingEnv import DroneJammingEnv

# Create and wrap the environment
env = DroneJammingEnv()

# Define the model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_drone")
```

## Rendering the trained agent
```python
import cv2
from DroneJammingEnv import DroneJammingEnv
from stable_baselines3 import PPO
import render  # Assuming render.py contains your rendering logic

# Load the trained model
model = PPO.load("ppo_drone")

# Create the environment
env = DroneJammingEnv()

# Capture frames and create a video
frames = render.capture_frames(env, model, num_frames=1000)
render.save_video(frames, 'drone_flight.avi')


