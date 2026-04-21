from dotenv import load_dotenv
load_dotenv()
from mapping.graphSLAM import FullGraphSLAMTracker, draw_graph_map
import os, sys, cv2
import math
import subprocess
import numpy as np

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")
sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot

MAX_SPEED    = 6.28
FORWARD_SPEED = MAX_SPEED * 0.55
TURN_SPEED    = MAX_SPEED * 0.35
WALL_MIN      = 0.12

os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"


def run_robot():
    # Launch Pioneer 3-AT controller as a separate process
    pioneer_script = os.path.join(os.path.dirname(__file__), "AutomaticMovement", "back_and_forth.py")
    pioneer_proc = subprocess.Popen([sys.executable, pioneer_script])

    os.environ["WEBOTS_CONTROLLER_URL"] = "TurtleBot3Burger"
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())

    left_motor  = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    left_ps  = robot.getDevice("left wheel sensor")
    right_ps = robot.getDevice("right wheel sensor")
    left_ps.enable(timestep)
    right_ps.enable(timestep)

    lidar = robot.getDevice("LDS-01")
    lidar.enable(timestep)

    tracker = FullGraphSLAMTracker(start_x=0.0, start_y=0.0, start_theta=0.0)

    # Warm up sensors
    for _ in range(5):
        robot.step(timestep)

    prev_left_rad  = left_ps.getValue()
    prev_right_rad = right_ps.getValue()

    for _ in range(5):
        robot.step(timestep)

    fov          = lidar.getFov()
    max_range    = lidar.getMaxRange()
    ranges_count = len(lidar.getRangeImage())

    print("Beginning GraphSLAM maze exploration")
    print("Manual control: W=forward  S=backward  A=turn left  D=turn right  Q=quit")

    left_speed  = 0.0
    right_speed = 0.0
    frame_counter = 0
    MAP_REDRAW_EVERY = 5   # only redraw the map every N Webots steps
    live_map = np.zeros((700, 700, 3), dtype=np.uint8)

    while robot.step(timestep) != -1:
        curr_left_rad  = left_ps.getValue()
        curr_right_rad = right_ps.getValue()

        delta_left  = curr_left_rad  - prev_left_rad
        delta_right = curr_right_rad - prev_right_rad

        prev_left_rad  = curr_left_rad
        prev_right_rad = curr_right_rad

        ranges = lidar.getRangeImage()
        if len(ranges) != ranges_count:
            ranges_count = len(ranges)

        # Build local scan
        current_scan_local = []
        angle_step = fov / (ranges_count - 1) if ranges_count > 1 else 0.0

        for i, distance in enumerate(ranges):
            angle_rad = (fov * 0.5) - (i * angle_step)
            local_x   = distance * math.cos(angle_rad)
            local_y   = distance * math.sin(angle_rad)
            if WALL_MIN < distance < max_range:
                current_scan_local.append((local_x, local_y))

        # GraphSLAM update (EKF + loop closure + optimization)
        tracker.add_odometry_and_scan(delta_left, delta_right, current_scan_local)
        frame_counter += 1

        heading_deg    = math.degrees(tracker.nodes[-1][2]) % 360
        num_nodes      = len(tracker.nodes)
        num_lc         = len(tracker.loop_closure_edges)

        # Keyboard control
        key = cv2.waitKey(30) & 0xFF
        if key == ord('w'):
            left_speed  = FORWARD_SPEED
            right_speed = FORWARD_SPEED
        elif key == ord('s'):
            left_speed  = -FORWARD_SPEED
            right_speed = -FORWARD_SPEED
        elif key == ord('a'):
            left_speed  = -TURN_SPEED
            right_speed =  TURN_SPEED
        elif key == ord('d'):
            left_speed  =  TURN_SPEED
            right_speed = -TURN_SPEED
        elif key == ord('q'):
            break
        else:
            left_speed  = 0.0
            right_speed = 0.0

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # Redraw map only every MAP_REDRAW_EVERY frames to avoid per-step cost
        if frame_counter % MAP_REDRAW_EVERY == 0:
            _, live_map = draw_graph_map(tracker, map_size=700, scale=100)

        # HUD overlay
        cv2.putText(live_map, f"Heading: {heading_deg:.1f} deg",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Nodes: {num_nodes}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(live_map, f"Loop closures: {num_lc}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1)

        cv2.imshow("Full GraphSLAM Map", live_map)

    cv2.destroyAllWindows()
    pioneer_proc.terminate()


if __name__ == "__main__":
    run_robot()
