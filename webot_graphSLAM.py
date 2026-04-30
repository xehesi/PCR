from dotenv import load_dotenv
load_dotenv()

from mapping.graphSLAM import GraphSession, draw_graph_map

import os, sys, cv2
import math
import subprocess
import numpy as np

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")
sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot

# ── TurtleBot3 Burger kinematics ─────────────────────────────────────────────
WHEEL_RADIUS = 0.033
WHEEL_BASE   = 0.160
# Sign flip used by the reference PCR stack to match Webots' yaw convention
# when odometry-integrating yaw. Unused when IMU yaw overrides theta.
ROTATION_SIGN = 0.89

# ── Manual control ───────────────────────────────────────────────────────────
MAX_SPEED     = 6.28
FORWARD_SPEED = MAX_SPEED * 0.85
TURN_SPEED    = MAX_SPEED * 0.55

# ── Sensor noise model (NOISE_ENABLED=False for ideal-sensor baseline) ──────
NOISE_ENABLED              = True
NOISE_SEED                 = 42
IMU_NOISE_SIGMA            = 0.01    # rad
LIDAR_NOISE_SIGMA          = 0.015   # m
ENCODER_COMMON_NOISE_SIGMA = 0.0040  # rad/step shared drift
ENCODER_DIFF_NOISE_SIGMA   = 0.0010  # rad/step differential (heading)
ENCODER_NOISE_MIN_MOTION   = 0.0010  # rad — suppress when stopped
ODOM_DELTA_DEADBAND        = 1e-5    # rad — ignore quantisation jitter

os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"


# ── Small helpers ────────────────────────────────────────────────────────────

def _build_lidar_angles(ray_count, lidar_fov):
    """Angle (rad, left = positive) for each lidar ray index — Webots sweep order."""
    if ray_count <= 0:
        return np.array([], dtype=float)
    return np.array(
        [
            (0.5 * lidar_fov) - ((ray_index + 0.5) * lidar_fov / float(ray_count))
            for ray_index in range(ray_count)
        ],
        dtype=float,
    )


# ── LiDAR-based dynamic ray filter (360°) ───────────────────────────────────
LIDAR_DYNAMIC_CHANGE_THRESHOLD = 0.15   # metres — must exceed expected ego-residual + sensor noise

def _lidar_dynamic_mask(
    prev_ranges,
    prev_pose,
    curr_ranges,
    curr_pose,
    lidar_angles,
    max_range,
    change_threshold=LIDAR_DYNAMIC_CHANGE_THRESHOLD,
):
    """Per-ray change detector that compensates for ego-motion.

    Projects the previous scan's body-frame points into the current body
    frame using the IMU-yaw + odometry delta, re-bins them by current ray
    index, and flags rays whose actual range differs from the predicted
    range by more than ``change_threshold``.

    Returns a boolean array of length ``len(curr_ranges)`` (True = dynamic).
    Falls back to all-False on the first tick or when ray-counts disagree.
    """
    n = len(curr_ranges)
    if prev_ranges is None or len(prev_ranges) != n or prev_pose is None:
        return np.zeros(n, dtype=bool)

    prev_r = np.asarray(prev_ranges, dtype=float)
    curr_r = np.asarray(curr_ranges, dtype=float)
    angles = np.asarray(lidar_angles, dtype=float)

    prev_valid = np.isfinite(prev_r) & (prev_r > 0.0) & (prev_r < max_range)
    if not prev_valid.any():
        return np.zeros(n, dtype=bool)

    # Body-frame xy of previous returns.
    px = prev_r * np.cos(angles)
    py = prev_r * np.sin(angles)

    # Transform prev body frame → current body frame.
    # rotation by (prev_theta - curr_theta), translation = prev origin
    # expressed in current body frame.
    dtheta = prev_pose[2] - curr_pose[2]
    c_d, s_d = math.cos(dtheta), math.sin(dtheta)
    dx_w = prev_pose[0] - curr_pose[0]
    dy_w = prev_pose[1] - curr_pose[1]
    c_c, s_c = math.cos(curr_pose[2]), math.sin(curr_pose[2])
    tx =  c_c * dx_w + s_c * dy_w
    ty = -s_c * dx_w + c_c * dy_w
    qx = c_d * px - s_d * py + tx
    qy = s_d * px + c_d * py + ty

    pred_range = np.hypot(qx, qy)
    pred_angle = np.arctan2(qy, qx)

    # Map predicted angles back to ray indices. Webots sweeps high → low,
    # so step is negative; handle either ordering generically.
    if n > 1:
        step = angles[1] - angles[0]
        if step == 0.0:
            return np.zeros(n, dtype=bool)
        idx_f = (pred_angle - angles[0]) / step
    else:
        idx_f = np.zeros_like(pred_angle)
    idx = np.round(idx_f).astype(int)
    in_range = (idx >= 0) & (idx < n) & prev_valid

    # Reduce: keep nearest predicted return per current ray bin.
    predicted = np.full(n, np.inf, dtype=float)
    for i in np.where(in_range)[0]:
        j = int(idx[i])
        if pred_range[i] < predicted[j]:
            predicted[j] = pred_range[i]

    curr_valid = np.isfinite(curr_r) & (curr_r > 0.0) & (curr_r < max_range)
    pred_valid = np.isfinite(predicted)
    both_valid = curr_valid & pred_valid
    diff = np.abs(curr_r - predicted)
    return both_valid & (diff > change_threshold)


def run_robot():
    try:
        pioneer_script = os.path.join(os.path.dirname(__file__),
                                      "AutomaticMovement", "back_and_forth.py")
        pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])
    except Exception:
        pioneer_proc = None

    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())

    # ── Motors ────────────────────────────────────────────────────────────
    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ── Wheel encoders ────────────────────────────────────────────────────
    left_ps  = robot.getDevice("left wheel sensor")
    right_ps = robot.getDevice("right wheel sensor")
    left_ps.enable(timestep)
    right_ps.enable(timestep)

    # ── LiDAR ─────────────────────────────────────────────────────────────
    lidar = robot.getDevice("LDS-01")
    lidar.enable(timestep)

    # ── Inertial unit ─────────────────────────────────────────────────────
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    # ── Noise RNG + IMU helper ────────────────────────────────────────────
    rng = np.random.default_rng(NOISE_SEED)

    def _get_imu_yaw():
        yaw = imu.getRollPitchYaw()[2]
        if NOISE_ENABLED:
            yaw += rng.normal(0.0, IMU_NOISE_SIGMA)
        return math.atan2(math.sin(yaw), math.cos(yaw))

    # ── Warm-up the sensors before reading odometry deltas ────────────────
    for _ in range(5):
        robot.step(timestep)

    prev_left_rad  = left_ps.getValue()
    prev_right_rad = right_ps.getValue()

    for _ in range(5):
        robot.step(timestep)

    lidar_fov       = lidar.getFov()
    lidar_max_range = lidar.getMaxRange()
    ranges_count    = len(lidar.getRangeImage())
    lidar_angles    = _build_lidar_angles(ranges_count, lidar_fov)

    # ── Odometry accumulator (encoder integration; IMU yaw overrides theta) ─
    odom_x = 0.0
    odom_y = 0.0
    odom_theta = _get_imu_yaw()

    # ── Previous-tick state for the LiDAR-based dynamic-ray filter ──────
    prev_lidar_ranges = None
    prev_lidar_pose   = None

    # ── Pose-graph session (tawan-slam logic) ─────────────────────────────
    session = GraphSession(
        node_dist_thresh=0.25,
        node_angle_thresh=math.radians(20),
        loop_radius=0.20,
        loop_warmup_nodes=8,
        icp_radius=0.35,
        icp_min_gap=8,
        icp_max_residual=0.06,
        icp_search_max_distance=0.5,
        lidar_fov=lidar_fov,
        lidar_max_range=lidar_max_range,
        odom_weight=1.0,
        loop_weight=5.0,
        optimize_iterations=10,
    )

    print("Beginning GraphSLAM maze exploration")
    print("Manual control: W=forward  S=backward  A=turn left  D=turn right  Q=quit")

    left_speed  = 0.0
    right_speed = 0.0
    frame_counter    = 0
    MAP_REDRAW_EVERY = 5
    live_map = np.zeros((700, 700, 3), dtype=np.uint8)

    while robot.step(timestep) != -1:
        # ── Wheel encoder deltas (with optional shared / differential noise) ─
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

        # ── IMU yaw (trusted as the robot's absolute heading) ────────────
        imu_yaw = _get_imu_yaw()

        # ── Odometry integration: encoder forward distance + IMU theta ───
        d_left   = delta_left  * WHEEL_RADIUS
        d_right  = delta_right * WHEEL_RADIUS
        d_center = 0.5 * (d_left + d_right)
        odom_theta = imu_yaw
        odom_x += d_center * math.cos(odom_theta)
        odom_y += d_center * math.sin(odom_theta)
        pose_vec = np.array([odom_x, odom_y, odom_theta], dtype=float)

        # ── Raw lidar scan + noise injection ─────────────────────────────
        ranges = lidar.getRangeImage()
        if NOISE_ENABLED:
            noisy = []
            for _r in ranges:
                if math.isinf(_r) or math.isnan(_r) or _r <= 0.0:
                    noisy.append(float('inf'))
                else:
                    noisy.append(max(0.01, min(lidar_max_range,
                                               _r + rng.normal(0.0, LIDAR_NOISE_SIGMA))))
            ranges = noisy
        if len(ranges) != ranges_count:
            ranges_count = len(ranges)
            lidar_angles = _build_lidar_angles(ranges_count, lidar_fov)

        # ── LiDAR-based 360° dynamic detection ───────────────────────────
        # Dynamic rays are set to inf so mapping.graphSLAM.scan_to_points
        # drops them (invalid) — moving objects never enter the SLAM map.
        lidar_dyn_mask = _lidar_dynamic_mask(
            prev_lidar_ranges, prev_lidar_pose,
            ranges, pose_vec,
            lidar_angles, lidar_max_range,
        )
        # Cache the *raw* (unfiltered) scan + pose for next tick's predictor.
        prev_lidar_ranges = list(ranges)
        prev_lidar_pose   = pose_vec.copy()

        scan_array = np.asarray(ranges, dtype=float).copy()
        scan_array[lidar_dyn_mask] = float('inf')

        # ── GraphSLAM update ─────────────────────────────────────────────
        optimized = session.step(pose_vec, scan_array)
        if optimized:
            tensions = session.last_tensions
            initial  = tensions[0]  if tensions else float('nan')
            final    = tensions[-1] if tensions else float('nan')
            loops    = ", ".join(f"({i}->{j})" for i, j in session.last_loop_edges)
            print(
                f"[graphSLAM] optimised: nodes={len(session.graph.nodes)} "
                f"edges={len(session.graph.edges)} "
                f"tension {initial:.4f} -> {final:.4f} "
                f"({len(tensions)} iters) loops=[{loops}]"
            )

        frame_counter += 1

        # ── Keyboard control ─────────────────────────────────────────────
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

        # ── Map render (throttled) ───────────────────────────────────────
        if frame_counter % MAP_REDRAW_EVERY == 0 and session.graph.nodes:
            _, live_map = draw_graph_map(session, map_size=700)

        # ── HUD overlay (display-only cv2 calls) ─────────────────────────
        num_nodes   = len(session.graph.nodes)
        num_lc      = sum(1 for e in session.graph.edges if e.edge_type == "loop")
        heading_deg = math.degrees(odom_theta) % 360
        cv2.putText(live_map, f"Heading: {heading_deg:.1f} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Nodes: {num_nodes}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Loop closures: {num_lc}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1)

        cv2.imshow("Full GraphSLAM Map", live_map)

    cv2.destroyAllWindows()
    if pioneer_proc is not None:
        pioneer_proc.terminate()


if __name__ == "__main__":
    run_robot()
