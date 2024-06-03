import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class DroneJammingEnv(gym.Env):
    def __init__(self, action_scale=5, connection_mode=p.GUI):
        super(DroneJammingEnv, self).__init__()
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.jamming_source = np.array([5, 5, 0])
        self.jamming_power = 10000

        self.time_step = 1.0 / 240.0
        self.max_steps = 10000
        p.setTimeStep(self.time_step)

        self.gravity = -10  # m/s^2, Gravity in the negative z-direction
        self.action_scale = action_scale  # Scaling factor for actions

        # Create the jamming source as a large red sphere
        self.jamming_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=5.0, rgbaColor=[1, 0, 0, 1])
        
        # Recreate the jamming source
        self.jamming_visual_body = p.createMultiBody(baseVisualShapeIndex=self.jamming_visual_shape, basePosition=self.jamming_source)
        #print("Jamming source created with body ID:", self.jamming_visual_body)
        
        # Zero out previous z velocity for stability reward 
        self.previous_z_velocity = 0  # Initialize previous z-velocity

        self.reset()

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()

        obs = self._get_obs()
        reward = self._calculate_signal_reward() + self._calculate_stability_reward()

        self.current_step += 1
        done = self.current_step > self.max_steps or self._is_drone_out_of_bounds()

        # Ensure the camera view is updated
        self.render()

        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf", basePosition=[0, 0, 0], globalScaling=1)

        # Randomize drone's initial position
        initial_x = np.random.randint(-5, 6)
        initial_y = np.random.randint(-5, 6)
        initial_z = np.random.randint(1, 11)
        self.drone = p.loadURDF("quadrotor.urdf", [initial_x, initial_y, initial_z])

        # Randomize jamming source position
        self.jamming_source = np.random.uniform(-50, 50, size=3)
        self.jamming_source[2] = 0  # Ensure it stays on the ground plane
        p.resetBasePositionAndOrientation(self.jamming_visual_body, self.jamming_source, [0, 0, 0, 1])

        self.current_step = 0
        
        # Reset previous signal to None at the start of each episode
        initial_obs = self._get_obs()
        self.previous_signal = initial_obs[-1]
        
        self.previous_z_velocity = 0  # Reset previous z-velocity at start of each episode
        
        return initial_obs

    def _apply_action(self, action):
        # Convert actions to thrust considering scaling
        thrust = (action * self.action_scale)
        dronePos, _ = p.getBasePositionAndOrientation(self.drone)

        # Ensure the drone cannot drop below 10 meters
        if dronePos[2] < 10:
            thrust[2] = max(thrust[2], self.gravity * -1)  # Upward thrust if below threshold

        p.applyExternalForce(self.drone, -1, thrust, dronePos, p.WORLD_FRAME)

    def _get_obs(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        drone_vel, _ = p.getBaseVelocity(self.drone)
        signal_strength = self._calculate_signal_strength(drone_pos)
        obs = np.concatenate([drone_pos, drone_vel, [signal_strength]])
        return obs

    def _calculate_signal_strength(self, drone_position):
        distance = np.linalg.norm(drone_position - self.jamming_source)
        signal_strength = self.jamming_power / (distance ** 2)
        return signal_strength

    def _calculate_signal_reward(self):
        current_position = p.getBasePositionAndOrientation(self.drone)[0]
        current_signal = self._calculate_signal_strength(current_position)
        if hasattr(self, 'previous_signal'):
            # Calculate the change in signal strength
            delta_signal = current_signal - self.previous_signal

            # Reward for increase in signal strength
            if delta_signal > 0:
                reward = delta_signal * 10  # Scale up the reward for positive changes
            else:
                # Penalize reduction in signal strength to encourage staying at the peak
                reward = delta_signal * 5  # More moderate scaling for negative changes
        else:
            reward = 0  # No previous signal to compare to at the start
        self.previous_signal = current_signal

        return reward


    def _calculate_stability_reward(self):
        _, drone_vel = p.getBaseVelocity(self.drone)
        # Penalize large velocities to encourage stability
        stability_reward = -np.linalg.norm(drone_vel)

        # Calculate the change in z-velocity and penalize large changes
        z_vel_change = drone_vel[2] - self.previous_z_velocity
        vertical_stability_reward = -10 * (z_vel_change ** 2)  # Scale the penalty as needed

        # Update previous z-velocity for next call
        self.previous_z_velocity = drone_vel[2]

        return stability_reward + vertical_stability_reward

    def _is_drone_out_of_bounds(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        x, y, z = drone_pos
        # Check if the drone is off the plane horizontally, below 0, or above 500
        if x < -1000 or x > 1000 or y < -1000 or y > 1000 or z < 0 or z > 500:
            print("Drone out of bounds:", drone_pos)
            return True
        return False

    def render(self, mode='human'):
        self._adjust_camera_view()

    def _adjust_camera_view(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        mid_point = (np.array(drone_pos) + np.array(self.jamming_source)) / 2
        max_distance = np.linalg.norm(np.array(drone_pos) - np.array(self.jamming_source))
        camera_distance = max_distance*0.9  # Increase camera distance for better visibility

        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=mid_point)
        #print("Camera adjusted to mid_point:", mid_point, "with distance:", camera_distance)

    def close(self):
        p.disconnect()

# Create an instance of the environment to test
#if __name__ == "__main__":
#    env = DroneJammingEnv()
#    env.reset()
