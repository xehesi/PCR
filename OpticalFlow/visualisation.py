"""
visualisation.py — Drawing utilities for optical flow and motion segmentation
==============================================================================

Provides helper functions that take the output of the optical flow pipeline and
render it onto an OpenCV BGR image for display inside a Webots simulation window.
"""

import cv2
import numpy as np


def draw_flow_field(
    bgr_frame: np.ndarray,
    points: np.ndarray,
    flows: np.ndarray,
    moving_mask: np.ndarray,
    arrow_scale: float = 4.0,
):
    """
    Draw optical flow arrows on `bgr_frame` in-place.

    Static-background points are drawn in GREEN; independently moving points
    are drawn in RED so they stand out immediately.

    Parameters
    ----------
    bgr_frame   : (H, W, 3) uint8 BGR image (modified in-place)
    points      : (N, 2) float — (x, y) grid locations
    flows       : (N, 2) float — (u, v) flow vectors
    moving_mask : (N,) bool — True for independently moving points
    arrow_scale : multiplier so short vectors are visible
    """
    for i in range(len(points)):
        x, y = int(points[i, 0]), int(points[i, 1])
        u, v = flows[i, 0], flows[i, 1]

        end_x = int(x + u * arrow_scale)
        end_y = int(y + v * arrow_scale)

        if moving_mask[i]:
            colour = (0, 0, 255)   # RED — independent motion
        else:
            colour = (0, 255, 0)   # GREEN — consistent with ego-motion

        cv2.arrowedLine(bgr_frame, (x, y), (end_x, end_y), colour, 1, tipLength=0.3)

    return bgr_frame


def draw_motion_regions(
    bgr_frame: np.ndarray,
    points: np.ndarray,
    moving_mask: np.ndarray,
    radius: int = 4,
):
    """
    Draw circles at grid positions flagged as independently moving.

    Parameters
    ----------
    bgr_frame   : (H, W, 3) uint8 BGR image (modified in-place)
    points      : (N, 2) float64
    moving_mask : (N,) bool
    radius      : circle radius in pixels
    """
    for i in range(len(points)):
        if moving_mask[i]:
            x, y = int(points[i, 0]), int(points[i, 1])
            cv2.circle(bgr_frame, (x, y), radius, (0, 0, 255), -1)
    return bgr_frame
