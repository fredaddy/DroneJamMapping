import pybullet as p
import pybullet_data
import time

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -10)

    # Create a large plane
    plane_path = pybullet_data.getDataPath() + "/plane.urdf"
    p.loadURDF(plane_path, basePosition=[0, 0, 0], globalScaling=1)

    # Create the jamming source as a large red sphere
    jamming_source = [0, 0, 10]
    jamming_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=20.0, rgbaColor=[1, 0, 0, 1])
    jamming_visual_body = p.createMultiBody(baseVisualShapeIndex=jamming_visual_shape, basePosition=jamming_source)
    print("Jamming source created with body ID:", jamming_visual_body)

    # Adjust the camera to see the jamming source
    p.resetDebugVisualizerCamera(cameraDistance=50, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 10])
    
    # Run the simulation for a while to observe the jamming source
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()

if __name__ == "__main__":
    main()
