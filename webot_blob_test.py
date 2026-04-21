"""
webot_blob_test.py — Target Detection + Lucas-Kanade Tracking
==============================================================

4-Phase pipeline:
  Phase 1: HSV color filter to isolate the target robot (background erased)
  Phase 2: Gaussian blur + Sobel edge detection on the filtered image,
           contour extraction, area-based sanity checks
  Phase 3: Compute the geometric center (u, v) of the best contour as a landmark
  Phase 4: Feed (u, v) to Lucas-Kanade for frame-to-frame tracking;
           suspend detection until the track is lost or its lifespan expires
"""

import os, sys, cv2
import numpy as np
import subprocess
from dotenv import load_dotenv
load_dotenv()

from image_processing.image_processing import (
    convolution, GAUSSIAN_KERNEL_3x3, GX, GY,
)
from OpticalFlow.optical_flow import rgb_to_grayscale, lucas_kanade_track

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")

sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot  # type: ignore

# ---- Speed ----
MAX_SPEED = 6.28
FORWARD_SPEED = MAX_SPEED * 0.55
TURN_SPEED = MAX_SPEED * 0.35
DISPLAY_SCALE = 3

# ---- Phase 1: HSV color bounds for the Pioneer 3-AT (red body) ----
# Red wraps around 0 in OpenCV HSV (H: 0-180), so two ranges are needed.
# Tune these if your target robot is a different colour.
# High saturation (>150) and value (>80) reject the dull floor/wall tones.
HSV_LOWER_1 = np.array([0, 150, 80])
HSV_UPPER_1 = np.array([10, 255, 255])
HSV_LOWER_2 = np.array([170, 150, 80])
HSV_UPPER_2 = np.array([180, 255, 255])

# ---- Phase 2: Contour area limits ----
MIN_CONTOUR_AREA = 80          # smaller = image noise / floor speckle
MAX_CONTOUR_AREA_FRAC = 0.50   # larger than 50 % of screen = artifact
EDGE_THRESHOLD = 50            # Sobel magnitude threshold for binary edges

# ---- Phase 4: Lucas-Kanade tracking ----
TRACK_LIFESPAN = 60            # re-detect after this many frames
LK_WINDOW_RADIUS = 7           # half-size of the LK tracking window


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
    # Gaussian blur (two passes) to smooth jagged mask edges
    mask_f = color_mask.astype(np.float32)
    blurred = convolution(mask_f, GAUSSIAN_KERNEL_3x3)
    blurred = convolution(blurred, GAUSSIAN_KERNEL_3x3)

    # Sobel edge detection on the blurred mask
    grad_x = convolution(blurred, GX)
    grad_y = convolution(blurred, GY)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Threshold to binary edges
    edge_binary = (edge_mag > EDGE_THRESHOLD).astype(np.uint8) * 255

    # Find contours from the edge image
    contours, _ = cv2.findContours(
        edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Area sanity checks
    max_area = img_area * MAX_CONTOUR_AREA_FRAC
    valid = [c for c in contours
             if MIN_CONTOUR_AREA < cv2.contourArea(c) < max_area]
    return valid, edge_binary


# ------------------------------------------------------------------
# Phase 3 — Landmark Generation
# ------------------------------------------------------------------
def phase3_landmark(contour):
    """Return the geometric centre (u, v) of a contour, or None."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def run_robot():
    # Launch the Pioneer 3-AT as a separate process
    pioneer_script = os.path.join(
        os.path.dirname(__file__), "AutomaticMovement", "back_and_forth.py")
    pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])

    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Motor setup
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # Camera setup
    camera = robot.getDevice("mycamera")
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()
    img_area = width * height

    prev_gray = None
    left_speed = 0.0
    right_speed = 0.0

    # Phase 4 state
    tracked_point = None   # (x, y) float tuple, or None
    track_age = 0

    print(f"Camera: {width}x{height}")
    print("Controls — W: forward | S: backward | A: left | D: right | Q: quit")

    while robot.step(timestep) != -1:
        image_array = get_image(camera, width, height)
        if image_array is None:
            continue

        curr_gray = rgb_to_grayscale(image_array)
        display = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # ==============================================================
        # Phase 4: Lucas-Kanade tracking (runs while we have a track)
        # ==============================================================
        if tracked_point is not None and prev_gray is not None:
            new_pts, ok = lucas_kanade_track(
                prev_gray, curr_gray,
                [tracked_point],
                window_radius=LK_WINDOW_RADIUS,
            )
            track_age += 1

            if ok[0] and track_age < TRACK_LIFESPAN:
                tracked_point = (float(new_pts[0, 0]), float(new_pts[0, 1]))
                cx, cy = int(tracked_point[0]), int(tracked_point[1])

                cv2.circle(display, (cx, cy), 6, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"TRACKING ({cx},{cy}) age={track_age}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                )
            else:
                # Track lost or lifespan expired — fall back to detection
                tracked_point = None
                track_age = 0

        # ==============================================================
        # Phases 1-3: Detection (only when not tracking)
        # ==============================================================
        if tracked_point is None:
            # Phase 1: Color filter
            color_mask = phase1_color_filter(image_array)

            # Phase 2: Edge-based contour extraction + area filter
            valid_contours, edge_vis = phase2_contour_extraction(
                color_mask, img_area)

            # Phase 3: Pick the largest valid contour as the landmark
            best_center = None
            best_contour = None
            best_area = 0
            for c in valid_contours:
                area = cv2.contourArea(c)
                if area > best_area:
                    center = phase3_landmark(c)
                    if center is not None:
                        best_center = center
                        best_contour = c
                        best_area = area

            if best_center is not None:
                cx, cy = best_center
                # Phase 4 init: start tracking from this landmark
                tracked_point = (float(cx), float(cy))
                track_age = 0

                cv2.drawContours(display, [best_contour], -1, (0, 255, 255), 1)
                cv2.circle(display, (cx, cy), 6, (0, 0, 255), 2)
                cv2.putText(
                    display,
                    f"DETECTED ({cx},{cy})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                )
            else:
                cv2.putText(
                    display, "Searching...",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                )

        # Always show the color mask so HSV bounds can be tuned
        if tracked_point is not None:
            color_mask = phase1_color_filter(image_array)
        mask_display = cv2.resize(
            color_mask,
            (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Color Mask (HSV)", mask_display)

        # Main display
        display = cv2.resize(
            display,
            (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Target Detection + Tracking", display)

        prev_gray = curr_gray.copy()

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