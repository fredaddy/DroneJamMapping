import numpy as np
import cv2
import pybullet as p

def capture_frames(env, model, num_frames=1000):
    frames = []
    obs = env.reset()
    done = False
    for _ in range(num_frames):
        if done:
            break
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
    return frames

def save_video(frames, filename, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
