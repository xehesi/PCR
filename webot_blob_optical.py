"""
webot_blob_optical.py — Combined HSV Blob Detection + Optical Flow Controller
===============================================================================

Pipeline:
1. Phase 1: HSV color filter to isolate the target robot (background erased)
2. Phase 2: Gaussian blur + Sobel edge detection, contour extraction, area filter
3. Phase 3: Compute bounding box and geometric centre of each valid contour
4. For each detected contour region, compute optical flow (with frame skipping)
5. Draw contour outlines, flow arrows, and centre landmark on the display
"""

import os, sys, cv2
import numpy as np
import subprocess
from dotenv import load_dotenv
load_dotenv()

from image_processing.image_processing import (
    convolution, GAUSSIAN_KERNEL_3x3, GX, GY,
)

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")

sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot  # type: ignore

from OpticalFlow.optical_flow import compute_flow_in_region
from OpticalFlow.visualisation import draw_flow_field


MAX_SPEED = 6.28
FORWARD_SPEED = MAX_SPEED * 0.55
TURN_SPEED = MAX_SPEED * 0.35

DISPLAY_SCALE = 6

# ---- Phase 1: HSV color bounds for the Pioneer 3-AT (red body) ----
HSV_LOWER_1 = np.array([0, 150, 80])
HSV_UPPER_1 = np.array([10, 255, 255])
HSV_LOWER_2 = np.array([170, 150, 80])
HSV_UPPER_2 = np.array([180, 255, 255])

# ---- Phase 2: Contour area limits ----
MIN_CONTOUR_AREA = 80
MAX_CONTOUR_AREA_FRAC = 0.50
EDGE_THRESHOLD = 50

# ---- Optical flow parameters (applied per-contour region) ----
FLOW_GRID_STEP = 4
FLOW_PATCH_RADIUS = 3
FLOW_MAX_DISP = 3
BBOX_PAD = 4          # extra pixels around the contour bounding box
FRAME_SKIP = 4        # compare current frame with N frames ago


def get_image(camera, width, height):
    raw = camera.getImage()
    if raw is None:
        return None
    img = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


# ------------------------------------------------------------------
# Phase 1 — HSV Color Filter
# ------------------------------------------------------------------
def phase1_color_filter(rgb_image):
    """Convert to HSV and mask pixels matching the target robot's colour."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER_1, HSV_UPPER_1)
    mask2 = cv2.inRange(hsv, HSV_LOWER_2, HSV_UPPER_2)
    return mask1 | mask2


# ------------------------------------------------------------------
# Phase 2 — Contour Extraction & Area Filter
# ------------------------------------------------------------------
def phase2_contour_extraction(color_mask, img_area):
    """Blur + Sobel on the binary mask, find contours, discard bad sizes."""
    mask_f = color_mask.astype(np.float32)
    blurred = convolution(mask_f, GAUSSIAN_KERNEL_3x3)
    blurred = convolution(blurred, GAUSSIAN_KERNEL_3x3)

    grad_x = convolution(blurred, GX)
    grad_y = convolution(blurred, GY)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_binary = (edge_mag > EDGE_THRESHOLD).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = img_area * MAX_CONTOUR_AREA_FRAC
    valid = [c for c in contours
             if MIN_CONTOUR_AREA < cv2.contourArea(c) < max_area]
    return valid


# ------------------------------------------------------------------
# Phase 3 — Bounding box + centre
# ------------------------------------------------------------------
def contour_bbox(contour, width, height, pad=BBOX_PAD):
    """Return (x_min, y_min, x_max, y_max) clamped to image bounds."""
    x, y, w, h = cv2.boundingRect(contour)
    return (max(x - pad, 0), max(y - pad, 0),
            min(x + w + pad, width), min(y + h + pad, height))


def contour_centre(contour):
    """Return (cx, cy) or None if degenerate."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def run_robot():
    # Launch Pioneer 3-AT in a separate process
    pioneer_script = os.path.join(os.path.dirname(__file__),
                                  "AutomaticMovement", "back_and_forth.py")
    pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])

    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Camera
    camera = robot.getDevice("mycamera")
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()
    img_area = width * height

    frame_buffer = []
    left_speed = 0.0
    right_speed = 0.0

    print(f"Camera: {width}x{height}")
    print("Controls — W/S/A/D to move, Q to quit")

    while robot.step(timestep) != -1:
        curr_frame = get_image(camera, width, height)
        if curr_frame is None:
            continue

        # ---- Phase 1: HSV color filter ----
        color_mask = phase1_color_filter(curr_frame)

        # ---- Phase 2: Contour extraction + area filter ----
        valid_contours = phase2_contour_extraction(color_mask, img_area)

        # ---- Display: camera feed as base ----
        display = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)

        # Reference frame for optical flow (FRAME_SKIP steps ago)
        ref_frame = (frame_buffer[-FRAME_SKIP]
                     if len(frame_buffer) >= FRAME_SKIP else None)

        total_flow_pts = 0

        for contour in valid_contours:
            x_min, y_min, x_max, y_max = contour_bbox(
                contour, width, height)
            centre = contour_centre(contour)

            # Draw contour outline and centre
            cv2.drawContours(display, [contour], -1, (0, 255, 255), 1)
            if centre:
                cv2.circle(display, centre, 4, (0, 0, 255), 2)

            # ---- Optical flow inside the bounding box ----
            if ref_frame is not None and (x_max - x_min) >= 10 and (y_max - y_min) >= 10:
                points, flows = compute_flow_in_region(
                    ref_frame, curr_frame,
                    x_min, y_min, x_max, y_max,
                    grid_step=FLOW_GRID_STEP,
                    patch_radius=FLOW_PATCH_RADIUS,
                    max_displacement=FLOW_MAX_DISP,
                )

                if len(points) > 0:
                    total_flow_pts += len(points)
                    moving_mask = np.ones(len(points), dtype=bool)
                    draw_flow_field(display, points, flows, moving_mask,
                                    arrow_scale=6.0)

            # Draw bounding box
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max),
                          (0, 255, 255), 1)

        label = f"Contours: {len(valid_contours)}  Flow pts: {total_flow_pts}"
        cv2.putText(display, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        display = cv2.resize(
            display,
            (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Blob + Optical Flow", display)

        # Show color mask for HSV tuning
        mask_display = cv2.resize(
            color_mask,
            (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Color Mask (HSV)", mask_display)

        # Update frame buffer
        frame_buffer.append(curr_frame.copy())
        if len(frame_buffer) > FRAME_SKIP + 1:
            frame_buffer.pop(0)

        # Keyboard control
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("w"):
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
        else:
            left_speed = 0.0
            right_speed = 0.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

    cv2.destroyAllWindows()
    pioneer_proc.terminate()


if __name__ == "__main__":
    run_robot()
