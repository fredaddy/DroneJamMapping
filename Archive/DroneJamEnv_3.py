import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class DroneJammingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(DroneJammingEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # Actions: forward, backward, left, right, up, down
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Signal strength

        self.jamming_source = np.array([0, 0, 0])  # Jamming source at the origin
        self.drone_initial_position = np.array([1000, 0, 0])  # 1000 meters away on the x-axis
        self.drone_position = self.drone_initial_position.copy()

        self.connect_to_pybullet()
        self.reset()

    def connect_to_pybullet(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self):
        p.resetSimulation()
        self.drone = p.loadURDF("quadrotor.urdf", self.drone_initial_position)
        self.drone_position = self.drone_initial_position.copy()
        return self._get_obs()

    def step(self, action):
        move = np.array([0, 0, 0])
        if action == 0:
            move[0] += 10  # Move forward
        elif action == 1:
            move[0] -= 10  # Move backward
        elif action == 2:
            move[1] += 10  # Move right
        elif action == 3:
            move[1] -= 10  # Move left
        elif action == 4:
            move[2] += 10  # Move up
        elif action == 5:
            move[2] -= 10  # Move down

        self.drone_position += move
        p.resetBasePositionAndOrientation(self.drone, self.drone_position, [0, 0, 0, 1])

        obs = self._get_obs()
        reward = self.calculate_signal_strength()
        done = False  # Condition to end the episode can be added here
        return obs, reward, done, {}

    def _get_obs(self):
        # Observation is the signal strength
        return np.array([self.calculate_signal_strength()])

    def calculate_signal_strength(self):
        # Calculate signal strength based on the inverse square law
        distance = np.linalg.norm(self.drone_position - self.jamming_source)
        if distance == 0:
            return float('inf')  # To handle division by zero
        return 1 / (distance ** 2)

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Drone Position: {self.drone_position}, Signal Strength: {self.calculate_signal_strength()}")

    def close(self):
        p.disconnect()

# Testing the environment
env = DroneJammingEnv()
for _ in range(10):
    action = env.action_space.sample()  # Randomly sample an action
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
