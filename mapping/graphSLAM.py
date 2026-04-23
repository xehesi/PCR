"""
Full GraphSLAM implementation — Milestone 3.

Architecture
------------
The map is a directed graph:
  nodes : list[np.ndarray shape (3,)]   — robot poses [x, y, theta]
  edges : list[dict]                    — odometry + loop-closure constraints

No fixed-size grid is used for map storage.  The graph grows dynamically and
is entirely agnostic to the robot's starting position.

External-library policy
-----------------------
  Allowed  : numpy (basic math / array ops), math stdlib
  Prohibited: scipy, cv2 for geometry (findContours, drawContours, line,
              circle, arrowedLine …).  cv2.imshow / cv2.waitKey / cv2.putText
              live in the caller (webot_graphSLAM.py) and are display-only.
"""

import math
import numpy as np

# ── Tuning constants ─────────────────────────────────────────────────────────
ROTATION_SIGN                  = 1.0
LOOP_CLOSURE_SPATIAL_THRESH    = 0.25   # metres between poses to attempt match
LOOP_CLOSURE_SIGNATURE_THRESH  = 0.10   # mean absolute signature error
LOOP_CLOSURE_MIN_NODE_GAP      = 18     # ignore the N most-recent nodes
LOOP_CLOSURE_LOOKBACK          = 240    # search only a local revisit window
LOOP_CLOSURE_CHECK_EVERY_N     = 10     # run loop-closure every N new nodes
LOOP_CLOSURE_MAX_CANDIDATES    = 2      # max ICP calls per check round
LOOP_TRANSLATION_RESIDUAL_MAX  = 0.12   # reject ICP closures that disagree on xy
LOOP_YAW_SCAN_BLEND            = 0.30
OPTIMIZE_EVERY_N_NODES         = 50     # run optimizer after this many nodes
_MAX_ICP_POINTS                = 60     # subsample scans before ICP
_GN_ITERS                      = 8      # Gauss-Newton iterations per call
_GN_DAMPING                    = 1e-4   # LM-style diagonal damping
_IMU_PRIOR_WEIGHT              = 120.0  # information weight for per-node IMU yaw priors
SCAN_SIGNATURE_BINS            = 72
ODOM_INFO_MATRIX               = np.diag([220.0, 220.0, 180.0])
LOOP_INFO_MATRIX               = np.diag([140.0, 140.0, 600.0])


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))



def _blend_angles(primary_angle, secondary_angle, secondary_weight):
    secondary_weight = max(0.0, min(1.0, secondary_weight))
    primary_weight = 1.0 - secondary_weight
    x_val = primary_weight * math.cos(primary_angle) + secondary_weight * math.cos(secondary_angle)
    y_val = primary_weight * math.sin(primary_angle) + secondary_weight * math.sin(secondary_angle)
    return math.atan2(y_val, x_val)


def _relative_pose(pose_i, pose_j):
    dx = pose_j[0] - pose_i[0]
    dy = pose_j[1] - pose_i[1]
    cos_theta = math.cos(pose_i[2])
    sin_theta = math.sin(pose_i[2])
    return np.array([
        cos_theta * dx + sin_theta * dy,
        -sin_theta * dx + cos_theta * dy,
        _normalize_angle(pose_j[2] - pose_i[2]),
    ], dtype=float)



def _scan_descriptor(scan, num_bins=SCAN_SIGNATURE_BINS, far_range=2.0):
    """Nearest-range descriptor over angular bins, compatible with range-style matching."""
    bins = np.full(num_bins, far_range, dtype=float)
    for lx, ly in scan:
        angle = math.atan2(ly, lx) % (2.0 * math.pi)
        idx   = int(angle / (2.0 * math.pi) * num_bins) % num_bins
        bins[idx] = min(bins[idx], math.hypot(lx, ly))
    return bins


def _scan_signature_error(d1, d2):
    return float(np.mean(np.abs(d1 - d2)))


def _subsample_scan(pts, max_pts):
    if len(pts) <= max_pts:
        return pts
    idx = np.round(np.linspace(0, len(pts) - 1, max_pts)).astype(int)
    return [pts[i] for i in idx]


def _icp_2d(src_pts, dst_pts, max_iter=15, tol=1e-4, max_match_dist=0.8):
    """
    Minimal 2-D ICP — written from scratch.
    Nearest-neighbour matching is fully vectorised (no Python loops over pts).
    Returns (dx, dy, dtheta) or None.
    """
    src_pts = _subsample_scan(src_pts, _MAX_ICP_POINTS)
    dst_pts = _subsample_scan(dst_pts, _MAX_ICP_POINTS)
    if len(src_pts) < 5 or len(dst_pts) < 5:
        return None

    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iter):
        # Vectorised nearest-neighbour: (N_src, N_dst) distance matrix
        diff  = src[:, None, :] - dst[None, :, :]          # (Ns, Nd, 2)
        dists = np.einsum('ijk,ijk->ij', diff, diff)        # squared dists
        j_best    = np.argmin(dists, axis=1)
        min_dists = np.sqrt(dists[np.arange(len(src)), j_best])
        valid     = min_dists < max_match_dist

        if valid.sum() < 5:
            return None

        ms = src[valid]
        md = dst[j_best[valid]]

        c_src = ms.mean(axis=0)
        c_dst = md.mean(axis=0)
        H     = (ms - c_src).T @ (md - c_dst)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = c_dst - R @ c_src

        prev_src = src.copy()
        src      = (R @ src.T).T + t
        R_total  = R @ R_total
        t_total  = R @ t_total + t

        if np.linalg.norm(src - prev_src) < tol:
            break

    dtheta = math.atan2(R_total[1, 0], R_total[0, 0])
    return float(t_total[0]), float(t_total[1]), float(dtheta)


# ── Spatial hash ──────────────────────────────────────────────────────────────

class _SpatialHash:
    """
    Hash-grid for O(1) average-case proximity queries.
    Replaces the O(n) linear scan in the original loop-closure check.
    """

    def __init__(self, cell_size):
        self._cs   = cell_size
        self._grid = {}          # (cx, cy) -> list[int]  (node indices)

    def _cell(self, x, y):
        return (int(math.floor(x / self._cs)),
                int(math.floor(y / self._cs)))

    def insert(self, node_idx, x, y):
        c = self._cell(x, y)
        bucket = self._grid.get(c)
        if bucket is None:
            self._grid[c] = [node_idx]
        else:
            bucket.append(node_idx)

    def rebuild(self, nodes):
        self._grid = {}
        for i, pose in enumerate(nodes):
            self.insert(i, pose[0], pose[1])

    def query_radius(self, x, y, radius):
        """All node indices whose cell is within `radius` of (x, y)."""
        r_cells = int(math.ceil(radius / self._cs)) + 1
        cx0, cy0 = self._cell(x, y)
        result = []
        for dcx in range(-r_cells, r_cells + 1):
            for dcy in range(-r_cells, r_cells + 1):
                bucket = self._grid.get((cx0 + dcx, cy0 + dcy))
                if bucket:
                    result.extend(bucket)
        return result


# ── Gauss-Newton pose-graph optimizer (no scipy) ─────────────────────────────

def _edge_error_and_jacobians(pi, pj, z):
    """
    Error vector e and analytical Jacobians J_i, J_j for one edge.

    h(pi, pj) is the relative pose of pj in pi's frame.
    e = h(pi, pj) - z,  then e[2] is angle-normalised.
    """
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    ci = math.cos(pi[2])
    si = math.sin(pi[2])

    h = np.array([
        ci * dx + si * dy,
        -si * dx + ci * dy,
        _normalize_angle(pj[2] - pi[2]),
    ])
    e    = h - z
    e[2] = _normalize_angle(e[2])

    # Jacobian of h wrt pi
    J_i = np.array([
        [-ci, -si, -si * dx + ci * dy],
        [ si, -ci, -ci * dx - si * dy],
        [  0,   0,                 -1],
    ])
    # Jacobian of h wrt pj
    J_j = np.array([
        [ci, si, 0],
        [-si, ci, 0],
        [  0,  0, 1],
    ])
    return e, J_i, J_j


def _optimize_graph_gn(nodes, edges, n_iters=_GN_ITERS, damping=_GN_DAMPING, imu_yaws=None):
    """
    Gauss-Newton pose-graph optimizer — implemented from scratch.

    Node 0 is held fixed (the anchor).  Nodes 1..N-1 are jointly optimised
    by solving the normal equations H dx = -b at each iteration.

    H and b are built analytically from the edge Jacobians and information
    matrices; np.linalg.solve (LAPACK) handles the linear system.

    Complexity per iteration: O(E) for Jacobian assembly + O((3N)^3) for
    the linear solve.  Scales comfortably to ~300 nodes on a laptop CPU.
    """
    n = len(nodes)
    if n < 3:
        return nodes

    poses = [p.copy() for p in nodes]
    dim   = 3 * (n - 1)                 # node-0 is fixed → excluded

    for _ in range(n_iters):
        H = np.zeros((dim, dim))
        b = np.zeros(dim)

        for edge in edges:
            i, j = edge["from"], edge["to"]
            if i >= n or j >= n:
                continue

            e, J_i, J_j = _edge_error_and_jacobians(
                poses[i], poses[j], edge["measurement"]
            )
            Omega  = edge["information"]
            JiT_O  = J_i.T @ Omega
            JjT_O  = J_j.T @ Omega

            ri = 3 * (i - 1)
            rj = 3 * (j - 1)

            if i > 0:
                H[ri:ri+3, ri:ri+3] += JiT_O @ J_i
                b[ri:ri+3]          += JiT_O @ e
            if j > 0:
                H[rj:rj+3, rj:rj+3] += JjT_O @ J_j
                b[rj:rj+3]          += JjT_O @ e
            if i > 0 and j > 0:
                off = JiT_O @ J_j
                H[ri:ri+3, rj:rj+3] += off
                H[rj:rj+3, ri:ri+3] += off.T

        # IMU yaw priors — one angle-only constraint per node that has a measurement.
        # Node 0 is fixed (excluded from H), so it is skipped.
        if imu_yaws is not None:
            for node_idx, yaw_meas in enumerate(imu_yaws):
                if yaw_meas is None or node_idx == 0 or node_idx >= n:
                    continue
                ai = 3 * (node_idx - 1) + 2   # theta row in the (n-1)*3 system
                yaw_err = _normalize_angle(poses[node_idx][2] - yaw_meas)
                H[ai, ai] += _IMU_PRIOR_WEIGHT
                b[ai]     += _IMU_PRIOR_WEIGHT * yaw_err

        # Levenberg-Marquardt diagonal damping for numerical stability
        H[np.arange(dim), np.arange(dim)] += damping

        try:
            dx_flat = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break

        # Apply correction to all non-anchor poses
        for k in range(1, n):
            ri         = 3 * (k - 1)
            poses[k]   = poses[k] + dx_flat[ri:ri+3]
            poses[k][2] = _normalize_angle(poses[k][2])

        if np.linalg.norm(dx_flat) < 1e-4:
            break

    return poses


# ── Custom drawing primitives (zero cv2 geometry calls) ──────────────────────

def _draw_line(img, x0, y0, x1, y1, color):
    """Bresenham integer line — works for any slope."""
    h, w = img.shape[:2]
    dx = abs(x1 - x0);  sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0);  sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = err << 1
        if e2 > -dy:
            err -= dy;  x0 += sx
        if e2 < dx:
            err += dx;  y0 += sy


def _draw_circle(img, cx, cy, radius, color):
    """Filled circle via numpy boolean mask — O(radius²) pixels."""
    h, w   = img.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask   = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    img[y0:y1, x0:x1][mask] = color


def _draw_arrow(img, x0, y0, x1, y1, color, tip_len=8):
    """Line segment with a two-sided arrowhead at (x1, y1)."""
    _draw_line(img, x0, y0, x1, y1, color)
    angle = math.atan2(y1 - y0, x1 - x0)
    for a_off in (math.pi * 5 / 6, -math.pi * 5 / 6):
        ax = int(x1 + tip_len * math.cos(angle + a_off))
        ay = int(y1 + tip_len * math.sin(angle + a_off))
        _draw_line(img, x1, y1, ax, ay, color)


# ── EKF odometry smoother ─────────────────────────────────────────────────────

class _EKF:
    """EKF that smooths wheel-encoder odometry; supports an optional IMU yaw correction step."""

    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta], dtype=float)
        self.P     = np.eye(3) * 0.01
        self.Q     = np.diag([0.02, 0.02, 0.005])

    def predict(self, d_center, delta_theta):
        theta = self.state[2]
        mid   = theta + delta_theta / 2.0
        self.state[0] += d_center * math.cos(mid)
        self.state[1] += d_center * math.sin(mid)
        self.state[2]  = _normalize_angle(self.state[2] + delta_theta)
        G = np.array([
            [1, 0, -d_center * math.sin(mid)],
            [0, 1,  d_center * math.cos(mid)],
            [0, 0,  1],
        ])
        self.P = G @ self.P @ G.T + self.Q

    def correct(self, yaw_measurement, yaw_std=0.01):
        """Kalman update step fusing an IMU yaw measurement into the EKF state."""
        H = np.array([[0.0, 0.0, 1.0]])          # observation matrix  (1, 3)
        R = yaw_std * yaw_std                     # measurement variance (scalar)
        innovation = _normalize_angle(yaw_measurement - self.state[2])
        S = float((H @ self.P @ H.T)[0, 0]) + R  # innovation covariance (scalar)
        K = self.P @ H.T / S                     # Kalman gain           (3, 1)
        self.state = self.state + K[:, 0] * innovation
        self.state[2] = _normalize_angle(self.state[2])
        I_KH = np.eye(3) - K @ H                 # Joseph form for numerical stability
        self.P = I_KH @ self.P @ I_KH.T + K * R @ K.T

    @property
    def pose(self):
        return self.state.copy()


# ── Main tracker ──────────────────────────────────────────────────────────────

class FullGraphSLAMTracker:
    """
    Graph-based SLAM tracker.

    Data structures
    ---------------
    nodes : list[np.ndarray(3)]  — pose graph vertices  [x, y, theta]
    scans : list[list[(x, y)]]   — local LiDAR scan per node (static pts only)
    edges : list[dict]           — graph edges (odometry + loop-closure)

    The graph is unbounded; no fixed grid is allocated.  The robot's starting
    position is the implicit origin but the optimiser is coordinate-free.
    """

    def __init__(self, start_x=0.0, start_y=0.0, start_theta=0.0, signature_far_range=2.0):
        # Graph
        self.nodes              = [np.array([start_x, start_y, start_theta],
                                            dtype=float)]
        self.scans              = [[]]
        self.descriptors        = [np.full(SCAN_SIGNATURE_BINS, signature_far_range, dtype=float)]
        self.edges              = []
        self.loop_closure_edges = []       # (i, j) pairs — visualisation only

        # Kinematics
        self.WHEEL_RADIUS = 0.033
        self.WHEEL_BASE   = 0.160
        self._signature_far_range = float(signature_far_range)

        # EKF + spatial hash
        self._ekf          = _EKF(start_x, start_y, start_theta)
        self._spatial_hash = _SpatialHash(cell_size=LOOP_CLOSURE_SPATIAL_THRESH)
        self._spatial_hash.insert(0, start_x, start_y)

        # IMU yaw measurements per node (None when no IMU available for that step)
        self._imu_yaws = [None]   # index 0 = start node; no IMU reading at init

        self._nodes_since_opt = 0
        self._nodes_since_lc  = 0

    # ── Main update ───────────────────────────────────────────────────────────

    def add_odometry_and_scan(self, delta_left_rad, delta_right_rad,
                               local_scan, imu_yaw=None, scan_signature=None):
        """
        Integrate one odometry step and one LiDAR scan into the graph.

        `local_scan` should contain only **static** obstacle points
        (dynamic objects pre-filtered by the caller).

        `imu_yaw` is an optional inertial-unit yaw reading (radians).  When
        provided it is fused into the EKF as a measurement update and stored
        as an angle prior for the pose-graph optimizer.
        """
        d_left      = delta_left_rad  * self.WHEEL_RADIUS
        d_right     = delta_right_rad * self.WHEEL_RADIUS
        d_center    = (d_right + d_left)  / 2.0
        delta_theta = ROTATION_SIGN * (d_right - d_left) / self.WHEEL_BASE

        self._ekf.predict(d_center, delta_theta)
        if imu_yaw is not None:
            self._ekf.correct(imu_yaw)
        new_pose = self._ekf.pose
        node_idx = len(self.nodes)
        prev_pose = self.nodes[node_idx - 1]
        odom_measurement = _relative_pose(prev_pose, new_pose)

        if scan_signature is None:
            descriptor = _scan_descriptor(
                local_scan,
                num_bins=SCAN_SIGNATURE_BINS,
                far_range=self._signature_far_range,
            )
        else:
            descriptor = np.asarray(scan_signature, dtype=float).copy()

        # Add vertex to graph
        self.nodes.append(new_pose)
        self.scans.append(local_scan)
        self.descriptors.append(descriptor)
        self._spatial_hash.insert(node_idx, new_pose[0], new_pose[1])
        self._imu_yaws.append(imu_yaw)   # None when no IMU; used by optimizer as angle prior

        # Sequential odometry edge
        self.edges.append({
            "from":        node_idx - 1,
            "to":          node_idx,
            "measurement": odom_measurement,
            "information": ODOM_INFO_MATRIX,
        })

        self._nodes_since_opt += 1
        self._nodes_since_lc  += 1

        if self._nodes_since_lc >= LOOP_CLOSURE_CHECK_EVERY_N:
            self._check_loop_closure(node_idx, local_scan)
            self._nodes_since_lc = 0

        if self._nodes_since_opt >= OPTIMIZE_EVERY_N_NODES:
            self._optimize_graph()
            self._nodes_since_opt = 0

        return self.nodes[node_idx]

    # ── Loop closure ──────────────────────────────────────────────────────────

    def _check_loop_closure(self, current_idx, current_scan):
        if current_idx < LOOP_CLOSURE_MIN_NODE_GAP + 1:
            return

        current_pose = self.nodes[current_idx]
        current_desc = self.descriptors[current_idx]
        earliest_idx = max(0, current_idx - LOOP_CLOSURE_LOOKBACK)

        # O(1) average — spatial hash replaces the old O(n) linear scan
        candidates = self._spatial_hash.query_radius(
            current_pose[0], current_pose[1], LOOP_CLOSURE_SPATIAL_THRESH
        )
        candidates.sort(
            key=lambda idx: math.hypot(
                self.nodes[idx][0] - current_pose[0],
                self.nodes[idx][1] - current_pose[1],
            )
        )

        icp_calls = 0
        for i in candidates:
            if i < earliest_idx or i >= current_idx - LOOP_CLOSURE_MIN_NODE_GAP:
                continue

            signature_error = _scan_signature_error(current_desc, self.descriptors[i])
            if signature_error > LOOP_CLOSURE_SIGNATURE_THRESH:
                continue

            if icp_calls >= LOOP_CLOSURE_MAX_CANDIDATES:
                break

            result = _icp_2d(current_scan, self.scans[i])
            icp_calls += 1
            if result is None:
                continue

            dx, dy, dtheta = result
            rel_pose = _relative_pose(self.nodes[i], current_pose)
            translation_residual = math.hypot(dx - rel_pose[0], dy - rel_pose[1])
            if translation_residual > LOOP_TRANSLATION_RESIDUAL_MAX:
                continue

            yaw_prior = rel_pose[2]
            if self._imu_yaws[i] is not None and self._imu_yaws[current_idx] is not None:
                yaw_prior = _normalize_angle(self._imu_yaws[current_idx] - self._imu_yaws[i])
            loop_yaw = _blend_angles(yaw_prior, dtheta, LOOP_YAW_SCAN_BLEND)

            self.edges.append({
                "from":        i,
                "to":          current_idx,
                "measurement": np.array([rel_pose[0], rel_pose[1], loop_yaw]),
                "information": LOOP_INFO_MATRIX,
            })
            self.loop_closure_edges.append((i, current_idx))
            print(f"[GraphSLAM] Loop closure: node {i} <-> {current_idx}"
                  f"  signature_error={signature_error:.3f}")


    # ── Graph optimisation ────────────────────────────────────────────────────

    def _optimize_graph(self):
        n = len(self.nodes)
        if n < 3:
            return

        updated      = _optimize_graph_gn(self.nodes, self.edges,
                                           imu_yaws=self._imu_yaws)
        self.nodes   = updated
        self._ekf.state = self.nodes[-1].copy()

        # Rebuild spatial hash with corrected positions
        self._spatial_hash.rebuild(self.nodes)

        print(f"[GraphSLAM] Optimized {n} nodes, {len(self.edges)} edges")


# ── Visualisation (display only — no map data stored here) ───────────────────

def draw_graph_map(tracker, map_size=700):
    """
    Render the graph to a numpy RGB image.

    Auto-scales to show all current nodes with a 10 % margin, so the display
    works regardless of the robot's starting position or environment size.

    Returns (background_img, display_img) — same interface as SLAM.py.
    """
    img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    if not tracker.nodes:
        return img, img.copy()

    # Compute bounding box from node positions; add padding for scan reach
    poses_arr = np.array([p[:2] for p in tracker.nodes])
    padding   = 2.0                        # metres — covers typical scan range
    min_x     = poses_arr[:, 0].min() - padding
    max_x     = poses_arr[:, 0].max() + padding
    min_y     = poses_arr[:, 1].min() - padding
    max_y     = poses_arr[:, 1].max() + padding

    world_w    = max(max_x - min_x, 0.1)
    world_h    = max(max_y - min_y, 0.1)
    margin     = 0.05
    auto_scale = min(map_size * (1.0 - margin) / world_w,
                     map_size * (1.0 - margin) / world_h)

    mid_x   = (min_x + max_x) / 2.0
    mid_y   = (min_y + max_y) / 2.0
    cx_px   = map_size // 2

    def to_px(wx, wy):
        return (int(cx_px + (wx - mid_x) * auto_scale),
                int(cx_px - (wy - mid_y) * auto_scale))

    # Draw scan points (grey) — vectorised per node
    for pose, scan in zip(tracker.nodes, tracker.scans):
        if not scan:
            continue
        cx, cy, ct = pose
        cos_t = math.cos(ct);  sin_t = math.sin(ct)
        sa    = np.array(scan, dtype=float)
        gx    = cx + sa[:, 0] * cos_t - sa[:, 1] * sin_t
        gy    = cy + sa[:, 0] * sin_t + sa[:, 1] * cos_t
        spx   = (cx_px + (gx - mid_x) * auto_scale).astype(int)
        spy   = (cx_px - (gy - mid_y) * auto_scale).astype(int)
        valid = (spx >= 0) & (spx < map_size) & (spy >= 0) & (spy < map_size)
        img[spy[valid], spx[valid]] = (200, 200, 200)

    # Trajectory (blue) — Bresenham lines between consecutive poses
    for i in range(1, len(tracker.nodes)):
        p1 = to_px(*tracker.nodes[i - 1][:2])
        p2 = to_px(*tracker.nodes[i][:2])
        _draw_line(img, p1[0], p1[1], p2[0], p2[1], (180, 80, 0))

    # Loop-closure edges (green)
    for (i, j) in tracker.loop_closure_edges:
        p1 = to_px(*tracker.nodes[i][:2])
        p2 = to_px(*tracker.nodes[j][:2])
        _draw_line(img, p1[0], p1[1], p2[0], p2[1], (0, 220, 0))

    display_img = img.copy()

    # Current robot pose: red dot + cyan heading arrow
    cx, cy, ct = tracker.nodes[-1]
    cpx, cpy   = to_px(cx, cy)
    if 0 <= cpx < map_size and 0 <= cpy < map_size:
        ax = int(cpx + 14 * math.cos(ct))
        ay = int(cpy - 14 * math.sin(ct))
        _draw_arrow(display_img, cpx, cpy, ax, ay, (0, 255, 255))
        _draw_circle(display_img, cpx, cpy, 5, (0, 0, 255))

    return img, display_img
