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

        self.jamming_source = np.array([50, 50, 0])
        self.jamming_power = 10000

        self.time_step = 1.0 / 240.0
        self.max_steps = 1e5
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
        #print(obs, reward)

        self.current_step += 1
        done = self.current_step > self.max_steps or self._is_drone_out_of_bounds()

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
        #self.jamming_source = np.random.uniform(-50, 50, size=3)
        #self.jamming_source[2] = 0  # Ensure it stays on the ground plane
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
        try:
            drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
            drone_vel, _ = p.getBaseVelocity(self.drone)
            signal_strength = self._calculate_signal_strength(drone_pos)
            obs = np.concatenate([drone_pos, drone_vel, [signal_strength]])
            return obs
        except p.error as e:
            print(f"Error in _get_obs: {e}")
            # Handle the error, e.g., return a default observation
            return np.zeros(self.observation_space.shape)


    def _calculate_signal_strength(self, drone_position):
        distance = np.linalg.norm(drone_position - self.jamming_source)
        signal_strength = self.jamming_power / (distance ** 2)
        return signal_strength

    def _calculate_signal_reward(self):
        current_position = p.getBasePositionAndOrientation(self.drone)[0]
        current_signal = self._calculate_signal_strength(current_position)
        
        # Reward based on signal strength
        reward = current_signal * 10  # Scale up the reward for high signal strength

        # Encourage staying close to the source
        distance_to_source = np.linalg.norm(current_position - self.jamming_source)
        if distance_to_source <= 11:  # Threshold distance to be considered "above" the source
            reward += 100  # Constant reward for being very close to the source
        
        # Penalize moving away from the source
        if hasattr(self, 'previous_signal'):
            delta_signal = current_signal - self.previous_signal
            if delta_signal < 0:
                reward += delta_signal * 50  # Higher penalty for decreasing signal strength
        self.previous_signal = current_signal

        return reward


    def _calculate_stability_reward(self):
        _, drone_vel = p.getBaseVelocity(self.drone)
        stability_reward = -np.linalg.norm(drone_vel)

        z_vel_change = drone_vel[2] - self.previous_z_velocity
        vertical_stability_reward = -10 * (z_vel_change ** 2)  # Keep the penalty for vertical instability

        self.previous_z_velocity = drone_vel[2]

        return stability_reward + vertical_stability_reward


    def _is_drone_out_of_bounds(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        x, y, z = drone_pos
        # Check if the drone is off the plane horizontally, below 0, or above 100
        if x < -100 or x > 100 or y < -100 or y > 100 or z < 0 or z > 100:
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
