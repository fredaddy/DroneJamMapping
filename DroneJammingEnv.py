import gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from gym import spaces
from stable_baselines3 import PPO
import cv2

class DroneJammingEnv(gym.Env):
    def __init__(self):
        super(DroneJammingEnv, self).__init__()

        # Connection to PyBullet
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Load drone model
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 1])

        # Initialize jamming source
        self.jamming_source = np.array([10, 10, 1])
        self.jamming_power = 10_000  # 10 kilowatts

        # Set simulation parameters
        self.time_step = 1.0 / 240.0
        self.max_steps = 1000

        self.reset()

    def step(self, action):
        p.stepSimulation()
        self._apply_action(action)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        distance = np.linalg.norm(self.drone_position - self.jamming_source)
        signal_strength = self.jamming_power / (distance**2)
        noise = np.random.normal(0, 0.05 * signal_strength)
        signal_strength += noise
        reward = -distance

        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps or distance < 1.0

        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)

        # Reload the plane and drone
        p.loadURDF("plane.urdf")
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 1])

        # Initial drone state
        self.drone_position = np.array([0, 0, 1])
        self.drone_velocity = np.array([0, 0, 0])
        self.current_step = 0

        return self._get_obs()

    def _apply_action(self, action):
        action = np.clip(action, -1, 1)
        thrust = (action + 1) * 0.5 * 20  # Scale action to thrust
        p.applyExternalForce(self.drone, -1, [0, 0, thrust[0]], [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone, -1, [0, 0, thrust[1]], [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone, -1, [0, 0, thrust[2]], [0, 0, 0], p.WORLD_FRAME)
        p.applyExternalForce(self.drone, -1, [0, 0, thrust[3]], [0, 0, 0], p.WORLD_FRAME)

        self.drone_position, self.drone_velocity, _, _ = p.getBasePositionAndOrientation(self.drone)

    def _get_obs(self):
        drone_pos, drone_ori = p.getBasePositionAndOrientation(self.drone)
        drone_vel, _ = p.getBaseVelocity(self.drone)
        obs = np.concatenate([drone_pos, drone_vel])
        return obs

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.drone_position,
            distance=10,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )
        img_arr = p.getCameraImage(640, 480, view_matrix, proj_matrix)
        w, h, rgb, depth, seg = img_arr
        return np.reshape(rgb, (h, w, 4))

# Test the environment
env = DroneJammingEnv()
obs = env.reset()
done = False
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
    time.sleep(1.0 / 240.0)
