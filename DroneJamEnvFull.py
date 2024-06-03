import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class DroneJammingEnv(gym.Env):
    def __init__(self, action_scale=5):
        super(DroneJammingEnv, self).__init__()
        self.client = p.connect(p.GUI)
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
        self.jamming_visual_body = p.createMultiBody(baseVisualShapeIndex=self.jamming_visual_shape, basePosition=self.jamming_source)
        print("Jamming source created with body ID:", self.jamming_visual_body)

        self.drone = None

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

        # Create a large plane
        plane_path = pybullet_data.getDataPath() + "/plane.urdf"
        p.loadURDF(plane_path, basePosition=[0, 0, 0], globalScaling=1)

        # Randomize initial position of the drone within some bounds
        initial_x = np.random.uniform(-5, 5)
        initial_y = np.random.uniform(-5, 5)
        initial_z = np.random.uniform(1, 10)
        self.drone = p.loadURDF("quadrotor.urdf", [initial_x, initial_y, initial_z])

        # Recreate the jamming source
        self.jamming_visual_body = p.createMultiBody(baseVisualShapeIndex=self.jamming_visual_shape, basePosition=self.jamming_source)
        print("Drone and plane created, Jamming source recreated with body ID:", self.jamming_visual_body)

        self.current_step = 0

        # Ensure the camera view is updated
        self.render()

        return self._get_obs()

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
        signal_strength = self._calculate_signal_strength(p.getBasePositionAndOrientation(self.drone)[0])
        return signal_strength

    def _calculate_stability_reward(self):
        _, drone_vel = p.getBaseVelocity(self.drone)
        # Penalize large velocities to encourage stability
        stability_reward = -np.linalg.norm(drone_vel)
        return stability_reward

    def _is_drone_out_of_bounds(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        x, y, z = drone_pos
        # Check if the drone is off the plane horizontally, below 0, or above 500
        if x < -50 or x > 50 or y < -50 or y > 50 or z < 0 or z > 500:
            print("Drone out of bounds:", drone_pos)
            return True
        return False

    def render(self, mode='human'):
        self._adjust_camera_view()

    def _adjust_camera_view(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        mid_point = (np.array(drone_pos) + np.array(self.jamming_source)) / 2
        max_distance = np.linalg.norm(np.array(drone_pos) - np.array(self.jamming_source))
        camera_distance = max_distance * 2  # Increase camera distance for better visibility

        p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=mid_point)
        #print("Camera adjusted to mid_point:", mid_point, "with distance:", camera_distance)

    def close(self):
        p.disconnect()

# Create an instance of the environment to test
if __name__ == "__main__":
    env = DroneJammingEnv()
    env.reset()
