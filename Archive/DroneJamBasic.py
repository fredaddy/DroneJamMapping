
### DroneJamEnvFull.py

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

class QuadcopterEnv(gym.Env):
    def __init__(self):
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load quadcopter URDF
        self.quadcopter_id = p.loadURDF("quadrotor.urdf", [0, 0, 0], useFixedBase=False)
        
        # Define action space (throttle command)
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        
        # Define observation space (position, velocity)
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]), 
                                                high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), 
                                                dtype=np.float32)
        
    def step(self, action):
        # Apply thrust commands
        thrust_x, thrust_y, thrust_z = action
        p.setJointMotorControl2(self.quadcopter_id, 0, p.VELOCITY_CONTROL, targetVelocity=thrust_z * 20)
        p.setJointMotorControl2(self.quadcopter_id, 1, p.VELOCITY_CONTROL, targetVelocity=thrust_y * 20)
        p.setJointMotorControl2(self.quadcopter_id, 2, p.VELOCITY_CONTROL, targetVelocity=thrust_x * 20)
        
        # Step simulation
        p.stepSimulation()
        
        # Get quadcopter state
        pos, orn = p.getBasePositionAndOrientation(self.quadcopter_id)
        vel, ang_vel = p.getBaseVelocity(self.quadcopter_id)
        
        # Convert quaternion to euler angles
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        
        # Return observation, reward, done, info
        observation = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
        reward = pos[2]  # Reward is the z position
        done = pos[2] >= 10  # Terminate when quadcopter reaches z = 10
        info = {}
        return observation, reward, done, info
        
    def reset(self):
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load quadcopter URDF
        self.quadcopter_id = p.loadURDF("quadcopter.urdf", [0, 0, 0], useFixedBase=False)
        
        # Set initial position and orientation
        p.resetBasePositionAndOrientation(self.quadcopter_id, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        
        # Step simulation to stabilize quadcopter
        for _ in range(100):
            p.stepSimulation()
        
        # Get initial state
        pos, _ = p.getBasePositionAndOrientation(self.quadcopter_id)
        vel, _ = p.getBaseVelocity(self.quadcopter_id)
        
        return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
    
    def render(self, mode='human'):
        pass  # PyBullet handles rendering
    
    def close(self):
        p.disconnect()