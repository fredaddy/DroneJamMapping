import gym
import numpy as np
import OpenGL.GL as gl
import glfw
import random
import cv2

class DroneJammingEnv(gym.Env):
    def __init__(self):
        super(DroneJammingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 100]), high=np.array([2000, 2000, 300]), dtype=np.float32)
        self.state = np.array([0, 0, 100])
        self.jamming_source = np.array([1000, 0, 100])
        self.signal_max_strength = 1000
        self.viewer = None
        self.q_table = np.random.uniform(low=-1, high=1, size=(10, 10, 10, 6))  # Simplified state space

    def discretize_state(self, state):
        """ Convert continuous state into a discrete state """
        max_val = self.observation_space.high
        bins = [np.linspace(0, max_val[i], num=10) for i in range(3)]
        discrete_state = [int(np.digitize(state[i], bins[i])) - 1 for i in range(3)]
        return tuple(discrete_state)

    def step(self, action):
        move = np.array([0, 0, 0])
        if action == 0:   # Move forward (+x)
            move += np.array([100, 0, 0])
        elif action == 1: # Move backward (-x)
            move += np.array([-100, 0, 0])
        elif action == 2: # Move right (+y)
            move += np.array([0, 100, 0])
        elif action == 3: # Move left (-y)
            move += np.array([0, -100, 0])
        elif action == 4: # Move up (+z)
            move += np.array([0, 0, 10])
        elif action == 5: # Move down (-z)
            move += np.array([0, 0, -10])

        new_position = np.clip(self.state + move, self.observation_space.low, self.observation_space.high)
        self.state = new_position
        signal_strength = self.calculate_signal_strength(self.state)
        reward = signal_strength
        done = np.linalg.norm(self.state - self.jamming_source) < 100  # Consider done if within 100 units of the source
        return self.state, reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            if not glfw.init():
                return False
            self.viewer = glfw.create_window(800, 600, "Drone Simulation", None, None)
            if not self.viewer:
                glfw.terminate()
                return False
            glfw.make_context_current(self.viewer)
            gl.glEnable(gl.GL_DEPTH_TEST)  # Enable depth testing
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color to black

        glfw.poll_events()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Adjust camera
        gl.glTranslatef(-500, -500, -1200)  # Move the camera back to see the scene
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glPointSize(20)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(self.jamming_source[0], self.jamming_source[1], self.jamming_source[2])
        gl.glEnd()

        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glPointSize(20)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(self.state[0], self.state[1], self.state[2])
        gl.glEnd()

        glfw.swap_buffers(self.viewer)
        return True


    def calculate_signal_strength(self, position):
        distance = np.linalg.norm(position - self.jamming_source)
        return self.signal_max_strength / max(distance ** 2, 1)  # Avoid division by zero

    def close(self):
        if self.viewer:
            glfw.terminate()

    def reset(self):
        self.state = np.array([0, 0, 100])
        return self.state

def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    for e in range(episodes):
        state = env.reset()
        done = False
        discrete_state = env.discretize_state(state)
        action = np.argmax(env.q_table[discrete_state]) if np.random.rand() > epsilon else env.action_space.sample()

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_discrete_state = env.discretize_state(next_state)
            next_action = np.argmax(env.q_table[next_discrete_state]) if np.random.rand() > epsilon else env.action_space.sample()
            old_value = env.q_table[discrete_state + (action,)]
            next_max = env.q_table[next_discrete_state + (next_action,)]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            env.q_table[discrete_state + (action,)] = new_value
            discrete_state = next_discrete_state
            action = next_action

        if e % 10 == 0:
            print(f"Episode {e}: Done")


def capture_frame(buffer_width, buffer_height):
    gl.glReadBuffer(gl.GL_FRONT)
    pixels = gl.glReadPixels(0, 0, buffer_width, buffer_height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    if pixels is None or len(pixels) == 0:
        print("No pixels read from buffer")
        return None
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(buffer_height, buffer_width, 3)
    image = np.flip(image, axis=0)  # Flip image in the vertical axis
    print(f"Frame captured: {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def simulate_optimal_path(env, model, filename='optimal_path.mp4', total_frames=100):
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    window = glfw.create_window(800, 600, "Drone Simulation", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return
    glfw.make_context_current(window)

    # Prepare video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    out = cv2.VideoWriter(filename, fourcc, 20.0, (800, 600))
    if not out.isOpened():
        print("Video writer could not be opened")
        return

    state = env.reset()
    done = False
    frame_count = 0

    while not done and frame_count < total_frames:
        discrete_state = env.discretize_state(state)
        action = np.argmax(env.q_table[discrete_state])  # Select the best action based on the learned policy
        state, _, done, _ = env.step(action)

        if not env.render():
            print("Render failed")
            break

        image = capture_frame(800, 600)
        if image is not None:
            out.write(image)
        else:
            print("Failed to capture frame")

        frame_count += 1
        percentage_complete = (frame_count / total_frames) * 100
        print(f"\rProgress: {percentage_complete:.2f}%", end='')

        glfw.swap_buffers(window)
        if glfw.window_should_close(window):
            print("Window should close")
            break

    out.release()
    glfw.terminate()

# Usage
env = DroneJammingEnv()
sarsa(env, 100)  # Assuming 'sarsa' function and SARSA training is already implemented
simulate_optimal_path(env, env.q_table)  # env.q_table should contain the learned Q-values



# Example of using the function
env = DroneJammingEnv()
sarsa(env, 100)  # Assuming 'sarsa' function and SARSA training is already implemented
simulate_optimal_path(env, env.q_table)  # env.q_table should contain the learned Q-values