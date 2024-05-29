
### DroneJamEnvFull.py

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

class DroneJammingEnv(gym.Env):
    def __init__(self):
        super(DroneJammingEnv, self).__init__()

        # Connection to PyBullet
        try:
            p.disconnect()
        except:
            pass
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define action and observation spaces
        # [dx, dy, dz, flag]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # [x, y, z, vx, vy, vz, signal_strength]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Load drone model
        self.drone = p.loadURDF("./quadrotor.urdf", [0, 0, 0])

        # Initialize jamming source to be random
        #self.jamming_source = np.random.rand(3) * 10 + 5

        self.jamming_source = np.array([5, 5, 1])
        self.jamming_power = 10_000  # 10 kilowatts
        self.signal_strength = 0 # to be added when steps.

        self.bomb_distance = 2.0
        self.dead_distance = 1.0

        # Set simulation parameters
        self.time_step = 1.0 / 240.0
        self.max_steps = 10000

        # Add signal jammer
        self.signal_jammer = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=1.0,
            rgbaColor=[1, 0, 0, 1],
            specularColor=[0.4, 0.4, 0],
        )
        self.signal_jammer_body = p.createMultiBody(
            baseVisualShapeIndex=self.signal_jammer,
            basePosition=self.jamming_source
        )

        self.destroyed = False
        self.correct_destroy = None
        self.incorrect_destroy = None

        self.reset()

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()

        # Adjust camera angle
        self.adjust_camera_angle()
        
        # Get observation
        obs = self._get_obs()

        done = False
        
        # Check if thing destroyed flag has been raised
        if self.destroyed:
            # If x-y coordinates are within bomb tolerance reward it
            if self.correct_destroy:
                reward = 1e6
                # And start returning to origin
                distance_to_origin = np.linalg.norm(self.drone_position)
                reward -= distance_to_origin
            # We blew up something random 
            else:
                reward = -1e8
                distance_to_origin = np.linalg.norm(self.drone_position)
                reward -= distance_to_origin
            if distance_to_origin < 0.1:
                    done = True
                    reward = 1e9

        else:
            reward = 0
            
            # Calculate Signal Reward
            distance = np.linalg.norm(self.drone_position - self.jamming_source)
            signal_strength = self.jamming_power / (distance ** 2)
            noise = np.random.normal(0, 0.1) * signal_strength
            signal_strength += noise
            self.signal_strength = signal_strength
    
            # Calculate Battery Reward
            #movement_penalty = np.linalg.norm(np.array(self.drone_position) - np.array(self.prev_drone_position))
            movement_penalty = self.current_step ** 1.5
            
            # Height Penalty
            if 0.1 <= self.drone_position[2] <= 5:
                height_reward = 1e3
            else:
                height_reward = -1e3
                done = True
    
            # Calculate Rewards
            reward_signal_strength = signal_strength
            reward_battery = -movement_penalty
            reward += reward_signal_strength + reward_battery + height_reward
        
            # Check if dead
            if distance < self.dead_distance:
                print("too close")
                reward += -1e6
                done = True

            # Check if out of bounds
            if abs(self.drone_position[0]) > 15:
                reward += -1e-2
                done = True

            if abs(self.drone_position[1]) > 15:
                reward += -1e-2
                done = True
    
            # Return to origin conditions
            if self.current_step >= self.max_steps - 1000:  # Start coming back if we're out for too long
                distance_to_origin = np.linalg.norm(self.drone_position)
                reward -= distance_to_origin 
            
            self.current_step += 1
            if self.current_step > self.max_steps:
                done = True
                if not self.correct_destroy:
                    reward += -1e9                    
                
            self.prev_drone_position = self.drone_position
        return obs, reward, done, {}

    def adjust_camera_angle(self):
        camera_position = [self.drone_position[0], self.drone_position[1], 5]
        target_position = self.drone_position
        up_vector = [0, 0, 1] 
    
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                           cameraTargetPosition=target_position,
                                           cameraUpVector=up_vector)
    
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-30,
                                      cameraTargetPosition=target_position)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.time_step)

        # Reload the plane and drone
        p.loadURDF("plane.urdf")
        self.drone = p.loadURDF("quadrotor.urdf", [0, 0, 0])

        # Initial drone state
        self.drone_position = np.array([0, 0, 0])
        self.prev_drone_position = np.array([0, 0, 0])
        self.drone_velocity = np.array([0, 0, 0])
        self.current_step = 0
        self.destroyed = False
        self.correct_destroy = None
        self.incorrect_destroy = None

        # Adjust camera parameters to control FOV
        camera_distance = 2.0  # Example value, adjust as needed
        camera_target_position = [0, 0, 0]  # Example position, adjust as needed
        camera_up_vector = [0, 0, 1]  # Example up vector, adjust as needed
        
        # Reset camera with adjusted parameters
        p.resetDebugVisualizerCamera(camera_distance, 45, -30, camera_target_position)

        return self._get_obs()

    def _apply_action(self, action):
        action = np.clip(action, -1, 1)
        thrust = (action + 1) * 10 # Scale action to thrust
        dronePos, droneOrn = p.getBasePositionAndOrientation(self.drone)
        p.applyExternalForce(self.drone, -1, thrust[:3], dronePos, p.WORLD_FRAME)

        # Flag for destroying the jammer, can only be turned on after step 20
        if (self.current_step > 20 and action[-1] == 1):
            self.destroyed = True
            dist = np.linalg.norm(self.drone_position - self.jamming_source)
            if dist < self.bomb_distance:
                self.correct_destroy = True
                self.incorrect_destroy = False
            else:
                self.correct_destroy = False
                self.incorrect_destroy = True
        self.drone_position, self.drone_velocity = p.getBasePositionAndOrientation(self.drone)

    def _get_obs(self):
        drone_pos, drone_ori = p.getBasePositionAndOrientation(self.drone)
        drone_vel, _ = p.getBaseVelocity(self.drone)
        obs = np.concatenate([drone_pos, drone_vel, [self.signal_strength]])
        return obs

    def render(self, mode='human'):
        p.resetDebugVisualizerCamera(
            cameraDistance=5, 
            cameraYaw=0, 
            cameraPitch=-30, 
            cameraTargetPosition=self.drone_position
        )

        img_arr = p.getCameraImage(640, 480)
        rgb = img_arr[2]

        return rgb
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=[0, 0, 0],  # Look at origin
        #     distance=50,
        #     yaw=90,  # Look along positive x-axis
        #     pitch=0,  # No pitch
        #     roll=0,  # No roll
        #     upAxisIndex=2
        # )

        # # Adjust the field of view (FOV) here
        # fov = 90  # Adjust FOV as needed
        # aspect_ratio = 640 / 480  # Adjust aspect ratio as needed
        # proj_matrix = p.computeProjectionMatrixFOV(
        #     fov=fov, aspect=aspect_ratio, nearVal=0.1, farVal=100.0
        # )

        # img_arr = p.getCameraImage(640, 480, view_matrix, proj_matrix)
        # rgb = img_arr[2]

        # return rgb
    def close(self):
        p.disconnect()