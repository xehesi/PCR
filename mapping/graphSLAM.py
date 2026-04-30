from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .icp import icp_match

def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def relative_pose(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Return pose of ``pj`` in ``pi``'s frame: [dx, dy, dtheta]."""
    dx = pj[0] - pi[0]
    dy = pj[1] - pi[1]
    c, s = math.cos(pi[2]), math.sin(pi[2])
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        wrap_angle(pj[2] - pi[2]),
    ])


def compose_se2(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Return world pose: ``pose`` composed with body-frame ``delta``."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    dx, dy, dth = float(delta[0]), float(delta[1]), float(delta[2])
    c, s = math.cos(th), math.sin(th)
    return np.array([
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        wrap_angle(th + dth),
    ])


# ── Graph primitives ─────────────────────────────────────────────────────────

@dataclass
class Node:
    id: int
    pose: np.ndarray            # [x, y, theta]
    scan: Optional[np.ndarray] = None   # raw lidar ranges


@dataclass
class Edge:
    i: int
    j: int
    measurement: np.ndarray     # [dx, dy, dtheta] of j in i's frame
    weight: float = 1.0
    edge_type: Literal["odom", "loop"] = "odom"


# ── Gauss-Newton optimiser ───────────────────────────────────────────────────

class GraphSLAM:
    """Pose-graph optimiser — numerical Jacobian, dense normal equations."""

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def poses(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros((0, 3))
        return np.stack([self.nodes[i].pose for i in sorted(self.nodes)], axis=0)

    def _calculate_error(self, pose_i, pose_j, measurement):
        xi, yi, theta_i = pose_i
        xj, yj, _ = pose_j
        dx = xj - xi
        dy = yj - yi
        c, s = math.cos(theta_i), math.sin(theta_i)
        pred = np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap_angle(pose_j[2] - theta_i),
        ])
        err = pred - measurement
        err[2] = wrap_angle(err[2])
        return err

    def _calculate_jacobian(self, pose_i, pose_j, measurement):
        eps = 1e-6
        base = self._calculate_error(pose_i, pose_j, measurement)
        J_i = np.zeros((3, 3))
        J_j = np.zeros((3, 3))
        for k in range(3):
            nudged = pose_i.copy();  nudged[k] += eps
            J_i[:, k] = (self._calculate_error(nudged, pose_j, measurement) - base) / eps
        for k in range(3):
            nudged = pose_j.copy();  nudged[k] += eps
            J_j[:, k] = (self._calculate_error(pose_i, nudged, measurement) - base) / eps
        return J_i, J_j, base

    def optimize(
        self,
        iterations: int = 10,
        anchor_weight: float = 1e6,
        verbose: bool = False,
    ) -> list[float]:
        """Gauss-Newton loop. Assumes node ids are contiguous from 0."""
        if not self.nodes or not self.edges:
            return []

        n = len(self.nodes)
        dim = n * 3
        tensions: list[float] = []

        for iteration in range(iterations):
            H = np.zeros((dim, dim))
            b = np.zeros(dim)
            total_tension = 0.0

            for edge in self.edges:
                pose_i = self.nodes[edge.i].pose
                pose_j = self.nodes[edge.j].pose
                J_i, J_j, err = self._calculate_jacobian(pose_i, pose_j, edge.measurement)
                total_tension += float(err @ err) * edge.weight

                w = edge.weight
                i3, j3 = edge.i * 3, edge.j * 3
                H[i3:i3+3, i3:i3+3] += (J_i.T @ J_i) * w
                H[i3:i3+3, j3:j3+3] += (J_i.T @ J_j) * w
                H[j3:j3+3, i3:i3+3] += (J_j.T @ J_i) * w
                H[j3:j3+3, j3:j3+3] += (J_j.T @ J_j) * w
                b[i3:i3+3] += J_i.T @ (err * w)
                b[j3:j3+3] += J_j.T @ (err * w)

            # Anchor node 0 (matches the reference: fixed by weight, not removed).
            for k in range(3):
                H[k, k] += anchor_weight

            try:
                movement = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(movement).all():
                break

            for i in range(n):
                self.nodes[i].pose[0] += movement[i*3 + 0]
                self.nodes[i].pose[1] += movement[i*3 + 1]
                self.nodes[i].pose[2] = wrap_angle(self.nodes[i].pose[2] + movement[i*3 + 2])

            tensions.append(total_tension)
            if verbose:
                print(f"[graphSLAM] iter {iteration+1}/{iterations} tension={total_tension:.4f}")

        return tensions


# ── LiDAR helpers ────────────────────────────────────────────────────────────

def scan_to_points(
    ranges,
    fov: float,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """Convert a 1-D lidar range array into (M, 2) body-frame points.

    Webots' Lidar sweeps from +fov/2 down to -fov/2. Invalid / over-range
    returns are dropped so caller-side dynamic masking (setting rays to inf)
    filters cleanly.
    """
    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0:
        return np.zeros((0, 2))
    n = ranges.size
    angles = np.linspace(fov / 2.0, -fov / 2.0, n) if n > 1 else np.array([0.0])
    valid = np.isfinite(ranges) & (ranges > 0)
    if max_range is not None:
        valid &= ranges < max_range
    r = ranges[valid]
    a = angles[valid]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


# ── Online session driver ────────────────────────────────────────────────────

class GraphSession:
    """Online pose-graph builder (keyframe-gated, ICP-loop-closed)."""

    def __init__(
        self,
        node_dist_thresh: float = 0.5,
        node_angle_thresh: float = math.radians(25),
        loop_radius: float = 0.15,
        loop_warmup_nodes: int = 8,
        icp_radius: float = 0.25,
        icp_min_gap: int = 8,
        icp_max_residual: float = 0.05,
        icp_search_max_distance: float = 0.5,
        lidar_fov: float = 2 * math.pi,
        lidar_max_range: Optional[float] = None,
        odom_weight: float = 1.0,
        loop_weight: float = 5.0,
        optimize_iterations: int = 10,
        min_anchor_translation: float = 0.02,
        allow_rotation_only_keyframes: bool = False,
    ) -> None:
        self.node_dist_thresh = node_dist_thresh
        self.node_angle_thresh = node_angle_thresh
        self.min_anchor_translation = min_anchor_translation
        self.allow_rotation_only_keyframes = allow_rotation_only_keyframes
        self.loop_radius = loop_radius
        self.loop_warmup_nodes = loop_warmup_nodes
        self.icp_radius = icp_radius
        self.icp_min_gap = icp_min_gap
        self.icp_max_residual = icp_max_residual
        self.icp_search_max_distance = icp_search_max_distance
        self.lidar_fov = lidar_fov
        self.lidar_max_range = lidar_max_range
        self.odom_weight = odom_weight
        self.loop_weight = loop_weight
        self.optimize_iterations = optimize_iterations

        self.graph = GraphSLAM()
        self._next_id: int = 0
        self._last_odom_pose: Optional[np.ndarray] = None
        self._last_node_id: Optional[int] = None
        self._scan_points_cache: dict[int, np.ndarray] = {}

        self.closed_to_start: bool = False
        self.optimized_this_tick: bool = False
        self.last_tensions: list[float] = []
        self.last_loop_edges: list[tuple[int, int]] = []
        self.pre_optim_poses: Optional[np.ndarray] = None

    # ---- helpers ----------------------------------------------------------

    def _should_add(self, odom: np.ndarray) -> bool:
        """Translation-first keyframe gate.

        A new keyframe is only inserted when the robot has translated past
        ``node_dist_thresh``. Rotation alone never triggers a keyframe by
        default — pure in-place rotation would otherwise stack near-duplicate
        scans at slightly drifting (x, y) (IMU + shared-mode encoder noise)
        and smear the rendered map. The angle threshold still fires as a
        secondary trigger, but only once the robot has *also* moved at least
        ``min_anchor_translation`` since the last keyframe (so curving paths
        get rotation-spaced keyframes; standing-and-spinning does not).

        Set ``allow_rotation_only_keyframes=True`` to recover the original
        tawan-slam behaviour for debugging.
        """
        if self._last_odom_pose is None:
            return True
        dx = odom[0] - self._last_odom_pose[0]
        dy = odom[1] - self._last_odom_pose[1]
        translation = math.hypot(dx, dy)
        dtheta = abs(wrap_angle(odom[2] - self._last_odom_pose[2]))

        if translation >= self.node_dist_thresh:
            return True
        if dtheta >= self.node_angle_thresh and (
            self.allow_rotation_only_keyframes
            or translation >= self.min_anchor_translation
        ):
            return True
        return False

    def _cached_points(self, node_id: int) -> Optional[np.ndarray]:
        if node_id in self._scan_points_cache:
            return self._scan_points_cache[node_id]
        node = self.graph.nodes.get(node_id)
        if node is None or node.scan is None:
            return None
        pts = scan_to_points(node.scan, self.lidar_fov, self.lidar_max_range)
        self._scan_points_cache[node_id] = pts
        return pts

    def _add_node(self, pose: np.ndarray, scan: Optional[np.ndarray]) -> int:
        nid = self._next_id
        self._next_id += 1
        stored = None if scan is None else np.asarray(scan, dtype=float).copy()
        self.graph.add_node(Node(id=nid, pose=pose.copy(), scan=stored))
        return nid

    # ---- main entry -------------------------------------------------------

    def step(self, pose: np.ndarray, scan: Optional[np.ndarray] = None) -> bool:
        """Feed one odometry pose + lidar scan. Returns True if re-optimised this tick."""
        odom = np.asarray(pose, dtype=float)
        self.optimized_this_tick = False
        self.last_loop_edges = []

        if self._last_odom_pose is None:
            nid = self._add_node(odom, scan)
            self._last_odom_pose = odom.copy()
            self._last_node_id = nid
            return False

        if not self._should_add(odom):
            return False

        prev_id = self._last_node_id
        assert prev_id is not None and self._last_odom_pose is not None

        delta = relative_pose(self._last_odom_pose, odom)
        prev_map = self.graph.nodes[prev_id].pose
        new_map = compose_se2(prev_map, delta)
        new_id = self._add_node(new_map, scan)
        self.graph.add_edge(Edge(
            i=prev_id, j=new_id,
            measurement=delta, weight=self.odom_weight,
            edge_type="odom",
        ))
        self._last_odom_pose = odom.copy()
        self._last_node_id = new_id

        loop_added = self._maybe_close_to_start(new_id)
        if scan is not None:
            loop_added = self._maybe_close_via_icp(new_id) or loop_added

        if loop_added:
            self.pre_optim_poses = self.graph.poses().copy()
            self.last_tensions = self.graph.optimize(iterations=self.optimize_iterations)
            self._scan_points_cache.clear()  # poses moved → cached world pts stale
            self.optimized_this_tick = True
            return True

        return False

    # ---- loop-closure strategies -----------------------------------------

    def _maybe_close_to_start(self, new_id: int) -> bool:
        if self.closed_to_start:
            return False
        if new_id <= self.loop_warmup_nodes:
            return False
        start = self.graph.nodes.get(0)
        if start is None:
            return False
        new_pose = self.graph.nodes[new_id].pose
        dx = new_pose[0] - start.pose[0]
        dy = new_pose[1] - start.pose[1]
        if math.hypot(dx, dy) >= self.loop_radius:
            return False
        self.graph.add_edge(Edge(
            i=new_id, j=0,
            measurement=np.zeros(3), weight=self.loop_weight,
            edge_type="loop",
        ))
        self.last_loop_edges.append((new_id, 0))
        self.closed_to_start = True
        return True

    def _maybe_close_via_icp(self, new_id: int) -> bool:
        new_pts = self._cached_points(new_id)
        if new_pts is None or new_pts.shape[0] < 3:
            return False

        new_pose = self.graph.nodes[new_id].pose
        loop_added = False
        cutoff_id = new_id - self.icp_min_gap
        for cand_id in range(0, cutoff_id + 1):
            cand = self.graph.nodes.get(cand_id)
            if cand is None or cand.scan is None:
                continue

            dist = math.hypot(new_pose[0] - cand.pose[0], new_pose[1] - cand.pose[1])
            if dist > self.icp_search_max_distance or dist > self.icp_radius:
                continue

            cand_pts = self._cached_points(cand_id)
            if cand_pts is None or cand_pts.shape[0] < 3:
                continue

            init = relative_pose(cand.pose, new_pose)
            z_icp, residual = icp_match(
                ref_pts=cand_pts, pts=new_pts,
                init=(float(init[0]), float(init[1]), float(init[2])),
            )
            if residual >= self.icp_max_residual:
                continue

            self.graph.add_edge(Edge(
                i=cand_id, j=new_id,
                measurement=z_icp, weight=self.loop_weight,
                edge_type="loop",
            ))
            self.last_loop_edges.append((cand_id, new_id))
            loop_added = True
            print(f"[graphSLAM] ICP loop closure: node {cand_id} <-> {new_id}"
                  f"  residual={residual:.3f}")

        return loop_added


# ── Drawing primitives (pure numpy Bresenham — no cv2 geometry) ─────────────

def _draw_line(img, x0, y0, x1, y1, color):
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
        if e2 > -dy:  err -= dy;  x0 += sx
        if e2 < dx:   err += dx;  y0 += sy


def _draw_circle(img, cx, cy, radius, color):
    h, w = img.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    img[y0:y1, x0:x1][mask] = color


def _draw_arrow(img, x0, y0, x1, y1, color, tip_len=8):
    _draw_line(img, x0, y0, x1, y1, color)
    angle = math.atan2(y1 - y0, x1 - x0)
    for a_off in (math.pi * 5 / 6, -math.pi * 5 / 6):
        ax = int(x1 + tip_len * math.cos(angle + a_off))
        ay = int(y1 + tip_len * math.sin(angle + a_off))
        _draw_line(img, x1, y1, ax, ay, color)


def draw_graph_map(session: GraphSession, map_size: int = 700):
    """Render the session's graph to a (H, W, 3) uint8 image.

    Auto-scales to fit all keyframes + their scan points with a margin. Returns
    (background_img, display_img) — callers overlay HUD / waitKey on display_img.
    """
    img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    graph = session.graph
    if not graph.nodes:
        return img, img.copy()

    node_ids = sorted(graph.nodes)
    poses_arr = np.stack([graph.nodes[i].pose[:2] for i in node_ids], axis=0)

    # Auto-fit bounds from poses + scan world points.
    scans_world: list[np.ndarray] = []
    for nid in node_ids:
        node = graph.nodes[nid]
        if node.scan is None:
            scans_world.append(np.zeros((0, 2)))
            continue
        body = scan_to_points(node.scan, session.lidar_fov, session.lidar_max_range)
        if body.shape[0] == 0:
            scans_world.append(body)
            continue
        c, s = math.cos(node.pose[2]), math.sin(node.pose[2])
        R = np.array([[c, -s], [s, c]])
        scans_world.append(body @ R.T + node.pose[:2])

    bounds_pts = [poses_arr]
    for pts in scans_world:
        if pts.shape[0] > 0:
            bounds_pts.append(pts)
    all_pts = np.concatenate(bounds_pts, axis=0)
    min_xy = all_pts.min(axis=0) - 0.5
    max_xy = all_pts.max(axis=0) + 0.5
    world_w = max(max_xy[0] - min_xy[0], 0.1)
    world_h = max(max_xy[1] - min_xy[1], 0.1)
    margin = 0.05
    scale = min(map_size * (1.0 - margin) / world_w,
                map_size * (1.0 - margin) / world_h)
    mid_x = (min_xy[0] + max_xy[0]) / 2.0
    mid_y = (min_xy[1] + max_xy[1]) / 2.0
    cx_px = map_size // 2

    def to_px(wx, wy):
        return (int(cx_px + (wx - mid_x) * scale),
                int(cx_px - (wy - mid_y) * scale))

    # Scan points (grey).
    for pts in scans_world:
        if pts.shape[0] == 0:
            continue
        spx = (cx_px + (pts[:, 0] - mid_x) * scale).astype(int)
        spy = (cx_px - (pts[:, 1] - mid_y) * scale).astype(int)
        valid = (spx >= 0) & (spx < map_size) & (spy >= 0) & (spy < map_size)
        img[spy[valid], spx[valid]] = (200, 200, 200)

    # Odometry trajectory (blue) + loop closures (green).
    odom_edges = [(e.i, e.j) for e in graph.edges if e.edge_type == "odom"]
    loop_edges = [(e.i, e.j) for e in graph.edges if e.edge_type == "loop"]
    for i, j in odom_edges:
        if i in graph.nodes and j in graph.nodes:
            p1 = to_px(*graph.nodes[i].pose[:2])
            p2 = to_px(*graph.nodes[j].pose[:2])
            _draw_line(img, p1[0], p1[1], p2[0], p2[1], (180, 80, 0))
    for i, j in loop_edges:
        if i in graph.nodes and j in graph.nodes:
            p1 = to_px(*graph.nodes[i].pose[:2])
            p2 = to_px(*graph.nodes[j].pose[:2])
            _draw_line(img, p1[0], p1[1], p2[0], p2[1], (0, 220, 0))

    display_img = img.copy()

    # Current pose marker (last-added node).
    last_pose = graph.nodes[node_ids[-1]].pose
    cpx, cpy = to_px(last_pose[0], last_pose[1])
    if 0 <= cpx < map_size and 0 <= cpy < map_size:
        ax = int(cpx + 14 * math.cos(last_pose[2]))
        ay = int(cpy - 14 * math.sin(last_pose[2]))
        _draw_arrow(display_img, cpx, cpy, ax, ay, (0, 255, 255))
        _draw_circle(display_img, cpx, cpy, 5, (0, 0, 255))

    return img, display_img
