import cv2

# Load the trained model
model = PPO.load("ppo_drone")

# Create the environment
env = DroneJammingEnv()

# Capture frames
frames = []
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    frame = env.render(mode='rgb_array')
    frames.append(frame)

# Save video
out = cv2.VideoWriter('drone_flight.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
