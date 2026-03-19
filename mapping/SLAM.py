import numpy as np
import math, cv2

ROTATION_SIGN = 0.89

class GraphSLAMTracker:
    def __init__(self, start_x=0.0, start_y=0.0, start_theta=0.0):
        self.nodes = [np.array([start_x, start_y, start_theta], dtype=float)]
        self.scans = [[]] 
        self.edges = []
        self.WHEEL_RADIUS = 0.033
        self.WHEEL_BASE = 0.160 

    def add_odometry_and_scan(self, delta_left_rad, delta_right_rad, local_scan):
        d_left = delta_left_rad * self.WHEEL_RADIUS
        d_right = delta_right_rad * self.WHEEL_RADIUS

        d_center = (d_right + d_left) / 2.0
        delta_theta = ROTATION_SIGN*(d_right - d_left) / self.WHEEL_BASE

        prev_pose = self.nodes[-1]
        theta_old = prev_pose[2]
        
        new_x = prev_pose[0] + d_center * math.cos(theta_old + (delta_theta / 2.0))
        new_y = prev_pose[1] + d_center * math.sin(theta_old + (delta_theta / 2.0))
        
        new_theta = theta_old + delta_theta
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

        new_pose = np.array([new_x, new_y, new_theta], dtype=float)
        
        node_idx = len(self.nodes)
        self.nodes.append(new_pose)
        self.scans.append(local_scan)
        
        constraint = {
            "from": node_idx - 1, 
            "to": node_idx, 
            "measurement": np.array([d_center, 0.0, delta_theta])
        } 
        self.edges.append(constraint)

        return new_pose

def draw_graph_map(tracker, background_img=None, map_size=600, scale=100, center_mode="origin"):
    if background_img is None:
        background_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        start_idx = 0
    else:
        start_idx = max(0, len(tracker.nodes) - 1)

    center_x, center_y = map_size // 2, map_size // 2
    anchor_x, anchor_y = 0.0, 0.0 

    for i in range(start_idx, len(tracker.nodes)):
        pose = tracker.nodes[i]
        scan = tracker.scans[i]
        curr_x, curr_y, curr_theta = pose[0], pose[1], pose[2]

        px = int(center_x + (curr_x - anchor_x) * scale)
        py = int(center_y - (curr_y - anchor_y) * scale)
        if 0 <= px < map_size and 0 <= py < map_size:
            cv2.circle(background_img, (px, py), 1, (0, 0, 120), -1)

        for (lx, ly) in scan:
            gx = curr_x + (lx * math.cos(curr_theta) - ly * math.sin(curr_theta))
            gy = curr_y + (lx * math.sin(curr_theta) + ly * math.cos(curr_theta))

            spx = int(center_x + (gx - anchor_x) * scale)
            spy = int(center_y - (gy - anchor_y) * scale)
            if 0 <= spx < map_size and 0 <= spy < map_size:
                background_img[spy, spx] = (220, 220, 220)

    display_img = background_img.copy()
    
    if len(tracker.nodes) > 0:
        curr_x, curr_y = tracker.nodes[-1][0], tracker.nodes[-1][1]
        curr_px = int(center_x + (curr_x - anchor_x) * scale)
        curr_py = int(center_y - (curr_y - anchor_y) * scale)
        if 0 <= curr_px < map_size and 0 <= curr_py < map_size:
            cv2.circle(display_img, (curr_px, curr_py), 5, (255, 0, 0), -1)


    return background_img, display_img