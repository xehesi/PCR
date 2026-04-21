import numpy as np
import math
import cv2
from scipy.optimize import least_squares

ROTATION_SIGN = 0.89

# ── Tuning constants ────────────────────────────────────────────────────────
LOOP_CLOSURE_SPATIAL_THRESH    = 1.0   # max metres between poses to attempt match
LOOP_CLOSURE_SIMILARITY_THRESH = 0.85  # cosine similarity threshold for scan descriptors
LOOP_CLOSURE_MIN_NODE_GAP      = 20    # ignore the N most-recent nodes when searching
LOOP_CLOSURE_CHECK_EVERY_N     = 10    # only run loop closure every N new nodes
LOOP_CLOSURE_MAX_CANDIDATES    = 3     # max ICP calls per loop closure check round
OPTIMIZE_EVERY_N_NODES         = 50    # run optimizer after this many new nodes
_MAX_ICP_POINTS                = 60    # subsample scans to this many points before ICP

# ── Low-level helpers ────────────────────────────────────────────────────────

def _normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _relative_pose(pose_i, pose_j):
    #Return the pose of j expressed in the frame of i
    dx = pose_j[0] - pose_i[0]
    dy = pose_j[1] - pose_i[1]
    ci = math.cos(pose_i[2])
    si = math.sin(pose_i[2])
    rel_x     =  ci * dx + si * dy
    rel_y     = -si * dx + ci * dy
    rel_theta = _normalize_angle(pose_j[2] - pose_i[2])
    return np.array([rel_x, rel_y, rel_theta])


def _scan_descriptor(scan, num_bins=36):
    #Encode a lidar scan as a mean-range histogram over angular bins.
    bins   = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    for (lx, ly) in scan:
        angle = math.atan2(ly, lx) % (2 * math.pi)
        idx   = int(angle / (2 * math.pi) * num_bins) % num_bins
        bins[idx]   += math.hypot(lx, ly)
        counts[idx] += 1
    counts[counts == 0] = 1
    return bins / counts


def _scan_similarity(d1, d2):
    #Cosine similarity between two scan descriptors.
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(d1, d2) / (n1 * n2))


def _subsample_scan(pts, max_pts):
    #Uniformly subsample a scan to at most max_pts points.
    if len(pts) <= max_pts:
        return pts
    idx = np.round(np.linspace(0, len(pts) - 1, max_pts)).astype(int)
    return [pts[i] for i in idx]


def _icp_2d(src_pts, dst_pts, max_iter=15, tol=1e-4, max_match_dist=0.8):
    #Minimal 2-D ICP.  Returns (dx, dy, dtheta) that aligns src onto dst, or None if matching fails. Inputs are subsampled before use.
    src_pts = _subsample_scan(src_pts, _MAX_ICP_POINTS)
    dst_pts = _subsample_scan(dst_pts, _MAX_ICP_POINTS)

    if len(src_pts) < 5 or len(dst_pts) < 5:
        return None

    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)

    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iter):
        # Nearest-neighbour matching
        matched_src, matched_dst = [], []
        for s in src:
            dists = np.linalg.norm(dst - s, axis=1)
            j = int(np.argmin(dists))
            if dists[j] < max_match_dist:
                matched_src.append(s)
                matched_dst.append(dst[j])

        if len(matched_src) < 5:
            return None

        ms = np.array(matched_src)
        md = np.array(matched_dst)

        c_src = ms.mean(axis=0)
        c_dst = md.mean(axis=0)

        H = (ms - c_src).T @ (md - c_dst)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Enforce proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = c_dst - R @ c_src

        prev_src = src.copy()
        src = (R @ src.T).T + t

        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.linalg.norm(src - prev_src) < tol:
            break

    dtheta = math.atan2(R_total[1, 0], R_total[0, 0])
    return float(t_total[0]), float(t_total[1]), float(dtheta)


# ── Extended Kalman Filter (single-pose, for odometry smoothing) ─────────────

class _EKF:
    #Predict-only EKF that smooths wheel-odometry noise before adding nodes.

    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta], dtype=float)
        self.P = np.eye(3) * 0.01
        # Process noise – increase if odometry is very noisy
        self.Q = np.diag([0.02, 0.02, 0.005])

    def predict(self, d_center, delta_theta):
        theta = self.state[2]
        mid   = theta + delta_theta / 2.0
        self.state[0] = self.state[0] + d_center * math.cos(mid)
        self.state[1] = self.state[1] + d_center * math.sin(mid)
        self.state[2] = _normalize_angle(self.state[2] + delta_theta)

        G = np.array([
            [1, 0, -d_center * math.sin(mid)],
            [0, 1,  d_center * math.cos(mid)],
            [0, 0,  1],
        ])
        self.P = G @ self.P @ G.T + self.Q

    @property
    def pose(self):
        return self.state.copy()


class FullGraphSLAMTracker:
    def __init__(self, start_x=0.0, start_y=0.0, start_theta=0.0):
        self.nodes             = [np.array([start_x, start_y, start_theta], dtype=float)]
        self.scans             = [[]]
        self.descriptors       = [np.zeros(36)]
        self.edges             = []
        self.loop_closure_edges = []   # list of (i, j) index pairs for visualisation
        self.WHEEL_RADIUS      = 0.033
        self.WHEEL_BASE        = 0.160
        self._ekf              = _EKF(start_x, start_y, start_theta)
        self._nodes_since_opt  = 0
        self._nodes_since_lc   = 0   # counts nodes since last loop closure check

    # ── Main update ──────────────────────────────────────────────────────────

    def add_odometry_and_scan(self, delta_left_rad, delta_right_rad, local_scan):
        d_left      = delta_left_rad  * self.WHEEL_RADIUS
        d_right     = delta_right_rad * self.WHEEL_RADIUS
        d_center    = (d_right + d_left) / 2.0
        delta_theta = ROTATION_SIGN * (d_right - d_left) / self.WHEEL_BASE

        # EKF smooths the raw wheel encoder noise
        self._ekf.predict(d_center, delta_theta)
        new_pose  = self._ekf.pose
        node_idx  = len(self.nodes)

        self.nodes.append(new_pose)
        self.scans.append(local_scan)
        self.descriptors.append(_scan_descriptor(local_scan))

        # Sequential odometry constraint
        self.edges.append({
            "from":        node_idx - 1,
            "to":          node_idx,
            "measurement": np.array([d_center, 0.0, delta_theta]),
            "information": np.diag([50.0, 50.0, 100.0]),
        })

        self._nodes_since_opt += 1
        self._nodes_since_lc  += 1

        # Only check loop closure every N nodes to avoid per-frame ICP cost
        if self._nodes_since_lc >= LOOP_CLOSURE_CHECK_EVERY_N:
            self._check_loop_closure(node_idx, local_scan)
            self._nodes_since_lc = 0

        if self._nodes_since_opt >= OPTIMIZE_EVERY_N_NODES:
            self._optimize_graph()
            self._nodes_since_opt = 0

        return self.nodes[node_idx]


    def _check_loop_closure(self, current_idx, current_scan):
        if current_idx < LOOP_CLOSURE_MIN_NODE_GAP + 1:
            return

        current_pose = self.nodes[current_idx]
        current_desc = self.descriptors[current_idx]

        icp_calls = 0
        for i in range(0, current_idx - LOOP_CLOSURE_MIN_NODE_GAP):
            candidate_pose = self.nodes[i]

            # 1. Spatial pre-filter (cheap – numpy-free hypot)
            dist = math.hypot(current_pose[0] - candidate_pose[0],
                              current_pose[1] - candidate_pose[1])
            if dist > LOOP_CLOSURE_SPATIAL_THRESH:
                continue

            # 2. Descriptor similarity (cheap dot product)
            sim = _scan_similarity(current_desc, self.descriptors[i])
            if sim < LOOP_CLOSURE_SIMILARITY_THRESH:
                continue

            # 3. ICP refinement (expensive – cap calls per round)
            if icp_calls >= LOOP_CLOSURE_MAX_CANDIDATES:
                break
            result = _icp_2d(current_scan, self.scans[i])
            icp_calls += 1
            if result is None:
                continue

            dx, dy, dtheta = result
            self.edges.append({
                "from":        i,
                "to":          current_idx,
                "measurement": np.array([dx, dy, dtheta]),
                "information": np.diag([200.0, 200.0, 400.0]),
            })
            self.loop_closure_edges.append((i, current_idx))
            print(f"[GraphSLAM] Loop closure: node {i} <-> node {current_idx}  (sim={sim:.2f})")


    def _optimize_graph(self):
        n = len(self.nodes)
        if n < 3:
            return

        # Node 0 is fixed; optimize nodes 1..n-1
        x0 = np.array([p.copy() for p in self.nodes[1:]], dtype=float).flatten()

        def residuals(x_flat):
            poses = [self.nodes[0]] + list(x_flat.reshape(-1, 3))
            res = []
            for edge in self.edges:
                i, j = edge["from"], edge["to"]
                if i >= n or j >= n:
                    continue
                sqrt_info = np.sqrt(np.diag(edge["information"]))
                e = _relative_pose(poses[i], poses[j]) - edge["measurement"]
                e[2] = _normalize_angle(e[2])
                res.extend(e * sqrt_info)
            return res

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=300, ftol=1e-4)
            optimized = result.x.reshape(-1, 3)
            for k, pose in enumerate(optimized):
                pose[2] = _normalize_angle(pose[2])
                self.nodes[k + 1] = np.array(pose, dtype=float)
            # Keep EKF in sync with the corrected latest pose
            self._ekf.state = self.nodes[-1].copy()
            print(f"[GraphSLAM] Graph optimized ({n} nodes, cost={result.cost:.4f})")
        except Exception as exc:
            print(f"[GraphSLAM] Optimization error: {exc}")


# ── Visualisation ────────────────────────────────────────────────────────────

def draw_graph_map(tracker, background_img=None, map_size=700, scale=100, center_mode="origin"):

    #Redraw the full map from scratch each frame so optimized poses stay correct. Returns (background_img, display_img) to match the SLAM.py interface.

    background_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    center_x, center_y = map_size // 2, map_size // 2
    anchor_x, anchor_y = 0.0, 0.0

    def to_px(wx, wy):
        px = int(center_x + (wx - anchor_x) * scale)
        py = int(center_y - (wy - anchor_y) * scale)
        return px, py

    # Scan points (grey)
    for pose, scan in zip(tracker.nodes, tracker.scans):
        cx, cy, ct = pose
        for (lx, ly) in scan:
            gx = cx + lx * math.cos(ct) - ly * math.sin(ct)
            gy = cy + lx * math.sin(ct) + ly * math.cos(ct)
            spx, spy = to_px(gx, gy)
            if 0 <= spx < map_size and 0 <= spy < map_size:
                background_img[spy, spx] = (200, 200, 200)

    # Trajectory (blue)
    for i in range(1, len(tracker.nodes)):
        p1 = to_px(*tracker.nodes[i - 1][:2])
        p2 = to_px(*tracker.nodes[i][:2])
        cv2.line(background_img, p1, p2, (180, 80, 0), 1)

    # Loop closure edges (green)
    for (i, j) in tracker.loop_closure_edges:
        p1 = to_px(*tracker.nodes[i][:2])
        p2 = to_px(*tracker.nodes[j][:2])
        cv2.line(background_img, p1, p2, (0, 220, 0), 1)

    display_img = background_img.copy()

    # Current robot pose (red dot + heading arrow)
    if tracker.nodes:
        cx, cy, ct = tracker.nodes[-1]
        cpx, cpy = to_px(cx, cy)
        if 0 <= cpx < map_size and 0 <= cpy < map_size:
            ax = int(cpx + 14 * math.cos(ct))
            ay = int(cpy - 14 * math.sin(ct))
            cv2.arrowedLine(display_img, (cpx, cpy), (ax, ay), (0, 255, 255), 2, tipLength=0.4)
            cv2.circle(display_img, (cpx, cpy), 5, (0, 0, 255), -1)

    return background_img, display_img
