import os, sys, cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
load_dotenv()
import time
import image_processing as ip
import blob_detection as bd
webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")

sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot # type: ignore

GAUSSIAN_KERNEL_3x3 = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32) / 16.0
GX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
GY = np.array([[1, 2, 1],
               [0, 0, 0],
                [-1, -2, -1]], dtype=np.float32)


def get_image(robot, width, height):
    camera = robot.getDevice("mycamera")
    raw_image = camera.getImage()
    if raw_image:
        image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        # Drop the Alpha channel
        image_array = image_array[:, :, :3]
        return image_array
    return None

def run_robot():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    previous_image = None
    
    camera = robot.getDevice("mycamera")
    camera.enable(timestep)
    
    width = camera.getWidth()
    height = camera.getHeight()

    print("Camera initialized. Press 'q' in either OpenCV window to exit.")

    while robot.step(timestep) != -1:
        image_array = get_image(robot, width, height)
        if image_array is None:
            continue

        blur = ip.gaussian_blur(Image.fromarray(image_array), GAUSSIAN_KERNEL_3x3)
        edge, grad_x, grad_y = ip.edge_detection_with_gradients(blur, GX, GY)

        all_blobs = bd.detect_blob(image_array, edge, grad_x, grad_y)
        
        target_blob = bd.find_ball_blob(all_blobs, width, height)

        if target_blob:
            blob_vis = bd.blobs_to_image([target_blob], width, height)
        else:
            blob_vis = np.zeros((height, width, 3), dtype=np.uint8)

        #original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) # Display the original
        blob_vis_bgr = cv2.cvtColor(blob_vis, cv2.COLOR_RGB2BGR)

        #cv2.imshow("Robot Vision (Raw Camera)", original_bgr) # Display the original camera feed
        cv2.imshow("Blob Detection (Target)", blob_vis_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_robot()