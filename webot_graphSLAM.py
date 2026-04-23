from dotenv import load_dotenv
load_dotenv()

from mapping.graphSLAM import FullGraphSLAMTracker, draw_graph_map
from image_processing.blob_detection import compute_motion_mask, get_dynamic_lidar_indices

import os, sys, cv2
import math
import subprocess
import numpy as np

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")
sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot

MAX_SPEED     = 6.28
FORWARD_SPEED = MAX_SPEED * 0.55
TURN_SPEED    = MAX_SPEED * 0.35
WALL_MIN      = 0.12

# Assume camera shares the robot's forward axis.
# Webots TurtleBot3 camera is typically 60° wide; getFov() returns this at runtime.
CAMERA_DEVICE_NAME = "mycamera"

# Sensor noise model (set NOISE_ENABLED = False for ideal-sensor baseline runs)
NOISE_ENABLED              = True
NOISE_SEED                 = 42
IMU_NOISE_SIGMA            = 0.01    # rad   — inertial unit yaw noise
LIDAR_NOISE_SIGMA          = 0.015   # m     — LiDAR range noise
ENCODER_COMMON_NOISE_SIGMA = 0.0040  # rad/step — shared drift (both wheels same direction)
ENCODER_DIFF_NOISE_SIGMA   = 0.0010  # rad/step — differential drift (heading error)
ENCODER_NOISE_MIN_MOTION   = 0.0010  # rad   — suppress noise when nearly stopped
ODOM_DELTA_DEADBAND        = 1e-5    # rad   — ignore encoder quantisation jitter
SCAN_SIGNATURE_BINS        = 72

os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"


def _to_gray(bgra_arr):
    """
    Convert a (H, W, 4) BGRA uint8 array to (H, W) uint8 greyscale.
    Luminosity weights applied in integer arithmetic — no cv2.cvtColor.
    """
    # Weights: R=0.299, G=0.587, B=0.114  (scaled to 1024 for integer speed)
    b = bgra_arr[:, :, 0].astype(np.uint32)
    g = bgra_arr[:, :, 1].astype(np.uint32)
    r = bgra_arr[:, :, 2].astype(np.uint32)
    return ((r * 306 + g * 601 + b * 117) >> 10).astype(np.uint8)


def _build_lidar_angles(ray_count, lidar_fov):
    if ray_count <= 0:
        return np.array([], dtype=float)
    return np.array(
        [
            (0.5 * lidar_fov) - ((ray_index + 0.5) * lidar_fov / float(ray_count))
            for ray_index in range(ray_count)
        ],
        dtype=float,
    )


def _build_scan_signature(ranges, bins, max_range, ignore_mask=None):
    values = np.asarray(ranges, dtype=float).copy()
    if ignore_mask is not None:
        values[np.asarray(ignore_mask, dtype=bool)] = float("inf")

    invalid = np.isnan(values) | np.isinf(values) | (values <= 0.0)
    values[invalid] = max_range
    values = np.clip(values, 0.0, max_range)

    signature = np.empty((bins,), dtype=float)
    ray_count = len(values)
    for idx in range(bins):
        start = int(idx * ray_count / bins)
        end = max(start + 1, int((idx + 1) * ray_count / bins))
        signature[idx] = float(np.median(values[start:end]))
    return signature


def run_robot():
    try:
        pioneer_script = os.path.join(os.path.dirname(__file__),
                                    "AutomaticMovement", "back_and_forth.py")
        pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])
    except Exception: # If the script fails to launch, we can still run the SLAM tracker without it.
        pass

    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ── Motors ────────────────────────────────────────────────────────────────
    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ── Wheel encoders ────────────────────────────────────────────────────────
    left_ps  = robot.getDevice("left wheel sensor")
    right_ps = robot.getDevice("right wheel sensor")
    left_ps.enable(timestep)
    right_ps.enable(timestep)

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    lidar = robot.getDevice("LDS-01")
    lidar.enable(timestep)

    # ── Inertial unit ─────────────────────────────────────────────────────────
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    # ── Camera (optional — graceful fallback if device absent) ───────────────
    camera     = robot.getDevice(CAMERA_DEVICE_NAME)
    has_camera = camera is not None
    if has_camera:
        camera.enable(timestep)
        cam_width  = camera.getWidth()
        cam_height = camera.getHeight()
        cam_fov    = camera.getFov()    # horizontal FOV in radians
    prev_gray  = None                   # previous greyscale frame

    # ── Noise RNG + IMU helper ────────────────────────────────────────────────
    rng = np.random.default_rng(NOISE_SEED)

    def _get_imu_yaw():
        yaw = imu.getRollPitchYaw()[2]
        if NOISE_ENABLED:
            yaw += rng.normal(0.0, IMU_NOISE_SIGMA)
        return math.atan2(math.sin(yaw), math.cos(yaw))

    # ── Sensor warm-up ────────────────────────────────────────────────────────
    for _ in range(5):
        robot.step(timestep)

    prev_left_rad  = left_ps.getValue()
    prev_right_rad = right_ps.getValue()

    for _ in range(5):
        robot.step(timestep)

    fov          = lidar.getFov()
    max_range    = lidar.getMaxRange()
    ranges_count = len(lidar.getRangeImage())

    # ── GraphSLAM tracker ─────────────────────────────────────────────────────
    tracker = FullGraphSLAMTracker(
        start_x=0.0,
        start_y=0.0,
        start_theta=0.0,
        signature_far_range=max_range,
    )

    # Pre-compute LiDAR angle for every ray index (radians, left = positive)
    lidar_angles = _build_lidar_angles(ranges_count, fov)

    print("Beginning GraphSLAM maze exploration")
    print("Manual control: W=forward  S=backward  A=turn left  D=turn right  Q=quit")

    left_speed  = 0.0
    right_speed = 0.0
    frame_counter   = 0
    MAP_REDRAW_EVERY = 5
    live_map = np.zeros((700, 700, 3), dtype=np.uint8)

    while robot.step(timestep) != -1:
        # ── Odometry ──────────────────────────────────────────────────────────
        curr_left_rad  = left_ps.getValue()
        curr_right_rad = right_ps.getValue()
        delta_left     = curr_left_rad  - prev_left_rad
        delta_right    = curr_right_rad - prev_right_rad
        prev_left_rad  = curr_left_rad
        prev_right_rad = curr_right_rad

        is_motion_step = (
            abs(delta_left) >= ODOM_DELTA_DEADBAND
            or abs(delta_right) >= ODOM_DELTA_DEADBAND
        )

        if NOISE_ENABLED and is_motion_step:
            motion = 0.5 * (abs(delta_left) + abs(delta_right))
            if motion > ENCODER_NOISE_MIN_MOTION:
                common       = rng.normal(0.0, ENCODER_COMMON_NOISE_SIGMA)
                differential = rng.normal(0.0, ENCODER_DIFF_NOISE_SIGMA)
                delta_left  += common - differential
                delta_right += common + differential

        # ── Inertial unit yaw ─────────────────────────────────────────────────
        imu_yaw = _get_imu_yaw()

        # ── LiDAR raw scan ────────────────────────────────────────────────────
        ranges = lidar.getRangeImage()
        if NOISE_ENABLED:
            noisy = []
            for _r in ranges:
                if math.isinf(_r) or math.isnan(_r) or _r <= 0.0:
                    noisy.append(float('inf'))
                else:
                    noisy.append(max(0.01, min(max_range,
                                               _r + rng.normal(0.0, LIDAR_NOISE_SIGMA))))
            ranges = noisy
        if len(ranges) != ranges_count:
            ranges_count = len(ranges)
            lidar_angles = _build_lidar_angles(ranges_count, fov)

        # ── Camera motion mask → dynamic-ray filter ───────────────────────────
        if has_camera:
            raw = camera.getImage()
            if raw:
                bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (cam_height, cam_width, 4)
                )
                curr_gray = _to_gray(bgra)

                if prev_gray is not None:
                    motion_mask   = compute_motion_mask(prev_gray, curr_gray,
                                                        threshold=20)
                    dynamic_mask  = get_dynamic_lidar_indices(
                        motion_mask, cam_fov, lidar_angles, cam_width
                    )
                else:
                    dynamic_mask = np.zeros(ranges_count, dtype=bool)

                prev_gray = curr_gray
            else:
                dynamic_mask = np.zeros(ranges_count, dtype=bool)
        else:
            dynamic_mask = np.zeros(ranges_count, dtype=bool)

        # ── Build static-only local scan ──────────────────────────────────────
        # Dynamic rays are excluded so moving objects never enter the SLAM map.
        static_scan = []
        for i, distance in enumerate(ranges):
            if dynamic_mask[i]:
                continue                          # moving object — skip
            if WALL_MIN < distance < max_range:
                angle_rad = lidar_angles[i]
                static_scan.append((distance * math.cos(angle_rad),
                                    distance * math.sin(angle_rad)))

        scan_signature = _build_scan_signature(
            ranges,
            bins=SCAN_SIGNATURE_BINS,
            max_range=max_range,
            ignore_mask=dynamic_mask,
        )

        # ── GraphSLAM update ──────────────────────────────────────────────────
        if is_motion_step:
            tracker.add_odometry_and_scan(
                delta_left,
                delta_right,
                static_scan,
                imu_yaw=imu_yaw,
                scan_signature=scan_signature,
            )
        frame_counter += 1

        heading_deg = math.degrees(tracker.nodes[-1][2]) % 360
        num_nodes   = len(tracker.nodes)
        num_lc      = len(tracker.loop_closure_edges)

        # ── Keyboard control ──────────────────────────────────────────────────
        key = cv2.waitKey(30) & 0xFF
        if key == ord('w'):
            left_speed, right_speed =  FORWARD_SPEED,  FORWARD_SPEED
        elif key == ord('s'):
            left_speed, right_speed = -FORWARD_SPEED, -FORWARD_SPEED
        elif key == ord('a'):
            left_speed, right_speed = -TURN_SPEED,  TURN_SPEED
        elif key == ord('d'):
            left_speed, right_speed =  TURN_SPEED, -TURN_SPEED
        elif key == ord('q'):
            break
        else:
            left_speed = right_speed = 0.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # ── Map redraw (throttled) ─────────────────────────────────────────────
        if frame_counter % MAP_REDRAW_EVERY == 0:
            _, live_map = draw_graph_map(tracker, map_size=700)

        # ── HUD overlay (display-only cv2 calls — permitted) ──────────────────
        cv2.putText(live_map, f"Heading: {heading_deg:.1f} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Nodes: {num_nodes}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Loop closures: {num_lc}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1)
        if has_camera:
            cv2.putText(live_map, "Sensor fusion: ON",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cv2.imshow("Full GraphSLAM Map", live_map)

    cv2.destroyAllWindows()
    pioneer_proc.terminate()


if __name__ == "__main__":
    run_robot()
