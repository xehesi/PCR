"""Lu & Milios style 2-D ICP.

Ported from https://github.com/richardos/icp but without sklearn:
nearest-neighbour search is a vectorised numpy argmin. Exposes ``icp_match``
which wraps ``icp`` with an initial-guess warm start and a Kabsch recovery of
the full (init + refinement) SE(2) transform plus a mean residual.
"""

from __future__ import annotations

import math
import numpy as np


def _nearest_neighbors(reference_points: np.ndarray, points: np.ndarray):
    """Vectorised (N_pts, N_ref) nearest-neighbour search."""
    diff = points[:, None, :] - reference_points[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(points.shape[0]), idx])
    return dist, idx


def point_based_matching(point_pairs):
    """Lu & Milios closed-form match: returns (rot_angle, tx, ty) or (None, None, None)."""
    n = len(point_pairs)
    if n == 0:
        return None, None, None

    x_mean = y_mean = xp_mean = yp_mean = 0.0
    for (x, y), (xp, yp) in point_pairs:
        x_mean += x;  y_mean += y
        xp_mean += xp;  yp_mean += yp
    x_mean /= n;  y_mean /= n
    xp_mean /= n;  yp_mean /= n

    s_x_xp = s_y_yp = s_x_yp = s_y_xp = 0.0
    for (x, y), (xp, yp) in point_pairs:
        s_x_xp += (x - x_mean) * (xp - xp_mean)
        s_y_yp += (y - y_mean) * (yp - yp_mean)
        s_x_yp += (x - x_mean) * (yp - yp_mean)
        s_y_xp += (y - y_mean) * (xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    tx = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
    ty = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))
    return rot_angle, tx, ty


def icp(
    reference_points: np.ndarray,
    points: np.ndarray,
    max_iterations: int = 40,
    distance_threshold: float = 0.3,
    convergence_translation_threshold: float = 1e-3,
    convergence_rotation_threshold: float = 1e-4,
    point_pairs_threshold: int = 10,
):
    """Align ``points`` (M, 2) onto ``reference_points`` (N, 2). Returns (history, aligned)."""
    history = []
    pts = np.asarray(points, dtype=float).copy()
    ref = np.asarray(reference_points, dtype=float)

    for _ in range(max_iterations):
        distances, indices = _nearest_neighbors(ref, pts)
        mask = distances < distance_threshold
        if int(mask.sum()) < point_pairs_threshold:
            break

        pairs = [(tuple(pts[i]), tuple(ref[indices[i]])) for i in np.where(mask)[0]]
        rot_angle, tx, ty = point_based_matching(pairs)
        if rot_angle is None or tx is None or ty is None:
            break

        c, s = math.cos(rot_angle), math.sin(rot_angle)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T + np.array([tx, ty])
        history.append(np.hstack((R, np.array([[tx], [ty]]))))

        if (abs(rot_angle) < convergence_rotation_threshold
                and abs(tx) < convergence_translation_threshold
                and abs(ty) < convergence_translation_threshold):
            break

    return history, pts


def icp_match(
    ref_pts: np.ndarray,
    pts: np.ndarray,
    init: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iterations: int = 40,
    distance_threshold: float = 0.3,
):
    """ICP with an initial-guess warm start; returns ([dx,dy,dtheta], residual)."""
    if ref_pts.shape[0] < 3 or pts.shape[0] < 3:
        return np.array(init, dtype=float), float("inf")

    dx, dy, dtheta = float(init[0]), float(init[1]), float(init[2])
    c, s = math.cos(dtheta), math.sin(dtheta)
    R_init = np.array([[c, -s], [s, c]])
    pts_warm = pts @ R_init.T + np.array([dx, dy])

    _, aligned = icp(
        ref_pts, pts_warm,
        max_iterations=max_iterations,
        distance_threshold=distance_threshold,
    )

    if aligned is None or aligned.shape[0] < 3:
        return np.array([dx, dy, dtheta]), float("inf")

    # Kabsch between original pts and fully-aligned output → total transform.
    pts_c = pts.mean(axis=0)
    aligned_c = aligned.mean(axis=0)
    H = (pts - pts_c).T @ (aligned - aligned_c)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.array([dx, dy, dtheta]), float("inf")
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = aligned_c - R @ pts_c
    total = np.array([t[0], t[1], math.atan2(R[1, 0], R[0, 0])])

    dist, _ = _nearest_neighbors(ref_pts, aligned)
    return total, float(dist.mean())
