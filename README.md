# DroneJamMapping
Reinforcement Learning module that trains a drone (agent) to identify point-source jammer autonomously and return coordinates to its operator. 

## Usage

```python
from stable_baselines3 import PPO

# Create and wrap the environment
env = DroneJammingEnv()

# Define the model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_drone")

