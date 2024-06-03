import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class DroneJammingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(DroneJammingEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # Corrected observation space definition to match the actual observations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.connect_to_pybullet()
        self.initialize_environment()

    def connect_to_pybullet(self):
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def initialize_environment(self):
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 100])
        self.drone_position = np.array([0, 0, 100])
        self.jamming_source = np.array([100, 100, 100])
        self.jamming_power = 10000
        self.create_signal_sphere(self.jamming_power)
        self.time_step = 1.0 / 240.0
        self.current_step = 0
        self.max_steps = 1000
        p.setTimeStep(self.time_step)

    def create_signal_sphere(self, initial_signal_strength):
        self.max_signal_radius = initial_signal_strength / 1000
        color = [1, 0, 0, 1]  # Solid red color with full opacity
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.max_signal_radius,
            rgbaColor=color,
            physicsClientId=self.client
        )
        body = p.createMultiBody(
            baseVisualShapeIndex=visual_shape,
            basePosition=self.jamming_source
        )

    def step(self, action):
        if self.drone_position[2] < 100:  # Keep the drone above 100 meters
            self.drone_position[2] = 100
        self._apply_action(action)
        p.stepSimulation()
        obs = self._get_obs()
        reward, done = self.calculate_rewards()
        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.initialize_environment()
        return self._get_obs()

    def _apply_action(self, action):
        action = np.clip(action, -1, 1)
        thrust = (action + 1) * 4
        dronePos, _ = p.getBasePositionAndOrientation(self.drone)
        p.applyExternalForce(self.drone, -1, thrust[:3], dronePos, p.WORLD_FRAME)
        self.drone_position, _ = p.getBasePositionAndOrientation(self.drone)

    def _get_obs(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        drone_pos = np.array(drone_pos)
        distance = np.linalg.norm(drone_pos - self.jamming_source)
        signal_strength = 1 / (distance ** 2) if distance != 0 else float('inf')
        return np.array([*drone_pos, signal_strength])

    def calculate_rewards(self):
        distance = np.linalg.norm(self.drone_position - self.jamming_source)
        signal_strength = self.jamming_power / (distance ** 2)
        reward = signal_strength - self.current_step * 0.1
        done = distance < 1.0 or self.current_step >= self.max_steps
        self.current_step += 1
        return reward, done

    def close(self):
        p.disconnect()
