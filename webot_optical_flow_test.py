
from dotenv import load_dotenv
load_dotenv()
import subprocess
import os
import sys
import subprocess
import cv2
import numpy as np
webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")
sys.path.append(os.path.join(webots_home, "lib", "controller", "python"))

from controller import Robot  # type: ignore
from OpticalFlow.optical_flow import process_frame_pair
from OpticalFlow.visualisation import draw_flow_field, draw_motion_regions

MAX_SPEED = 6.28
FORWARD_SPEED = MAX_SPEED * 0.55
TURN_SPEED = MAX_SPEED * 0.35

# Optical flow parameters
GRID_STEP = 4          # pixels between sampled grid points (denser = better at distance)
MAX_DISPLACEMENT = 3   # search bound for (u, v)
PATCH_RADIUS = 3       # half-size of neighbourhood (3 → 7×7 patch, 49 equations)
MOTION_THRESHOLD = 0.5 # flow magnitude above median to flag as moving
FRAME_SKIP = 6         # compare current frame with N frames ago (amplifies slow motion)
DISPLAY_SCALE = 3      # factor to enlarge the display window


def get_camera_image(camera, width, height):
    raw = camera.getImage()
    if raw is None:
        return None
    # Webots image → numpy (H, W, 4) BGRA
    img = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    # Convert BGRA → RGB and drop alpha
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def run_robot():
    """Main Webots control loop."""
    # Launch Pioneer 3-AT controller as a separate process
    pioneer_script = os.path.join(os.path.dirname(__file__), "AutomaticMovement", "back_and_forth.py")
    pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])
    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ---- Motor setup ----
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ---- Camera setup ----
    camera = robot.getDevice("mycamera")
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()

    # Keep a short buffer of past frames; compare against FRAME_SKIP frames ago
    # so slow-moving objects accumulate enough pixel displacement to be detected
    # by the integer-only MSE search.
    frame_buffer = []
    left_speed = 0.0
    right_speed = 0.0

    print(f"Camera: {width}x{height}")
    print("Controls — W: forward | S: backward | A: turn left | D: turn right | Q: quit")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while robot.step(timestep) != -1:
        # ----------------------------------------------------------
        # 1. Capture the current frame
        # ----------------------------------------------------------
        curr_frame = get_camera_image(camera, width, height)
        if curr_frame is None:
            continue

        # ----------------------------------------------------------
        # 2. Run optical flow & motion segmentation
        # ----------------------------------------------------------
        # Use the frame from FRAME_SKIP steps ago so slow body motion
        # accumulates enough pixel displacement to cross the integer
        # threshold of the MSE search.
        ref_frame = frame_buffer[-FRAME_SKIP] if len(frame_buffer) >= FRAME_SKIP else None

        if ref_frame is not None:
            points, flows, moving_mask = process_frame_pair(
                ref_frame, curr_frame,
                grid_step=GRID_STEP,
                patch_radius=PATCH_RADIUS,
                max_displacement=MAX_DISPLACEMENT,
                motion_threshold=MOTION_THRESHOLD,
            )

            # ----- Visualisation -----
            display = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)

            if len(points) > 0:
                # Draw flow arrows (green = static, red = moving)
                draw_flow_field(display, points, flows, moving_mask, arrow_scale=6.0)

                # Count independently-moving points for the HUD
                n_moving = int(np.sum(moving_mask))
                label = f"Moving pts: {n_moving}/{len(points)}"
            else:
                label = "No flow data"

            cv2.putText(display, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Scale the display up so the window is easier to see
            display = cv2.resize(
                display,
                (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("Optical Flow — Motion Segmentation", display)

        # Save current frame into the ring buffer
        frame_buffer.append(curr_frame.copy())
        if len(frame_buffer) > FRAME_SKIP + 1:
            frame_buffer.pop(0)

        # ----------------------------------------------------------
        # 3. Keyboard control (same scheme as other controllers)
        # ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("w"):
            left_speed = FORWARD_SPEED
            right_speed = FORWARD_SPEED
        elif key == ord("s"):
            left_speed = -FORWARD_SPEED
            right_speed = -FORWARD_SPEED
        elif key == ord("a"):
            left_speed = -TURN_SPEED
            right_speed = TURN_SPEED
        elif key == ord("d"):
            left_speed = TURN_SPEED
            right_speed = -TURN_SPEED
        elif key == ord("q"):
            break
        else:
            left_speed = 0.0
            right_speed = 0.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

    cv2.destroyAllWindows()
    pioneer_proc.terminate()


if __name__ == "__main__":
    run_robot()
