"""
optical_flow.py — Pure-NumPy Optical Flow via Brightness Constancy & MSE Minimization
======================================================================================

This module implements optical flow from scratch, following the pipeline from course
lectures on computer vision and motion estimation. No pre-built OpenCV optical flow
functions (e.g., calcOpticalFlowPyrLK) are used.

THEORETICAL BACKGROUND
----------------------
The Brightness Constancy Assumption states that the intensity of a pixel does not
change as it moves between consecutive frames:

    I(x, y, t) = I(x + u, y + v, t + 1)

where (u, v) is the displacement (optical flow) of that pixel. Taking a first-order
Taylor expansion and dividing by dt gives the **optical flow constraint equation**:

    dI/dx * u  +  dI/dy * v  +  dI/dt  ≈  0

This single equation is under-determined for two unknowns (u, v). We resolve this
by assuming that every pixel in a small 3×3 neighbourhood shares the same (u, v),
yielding 9 equations (one per pixel in the patch) and 2 unknowns.

PIPELINE
--------
1. **Preprocessing** — Convert to grayscale; apply Gaussian blur (3×3, σ ≈ 1) to
   suppress sensor noise.
2. **Spatial gradients** — Compute dI/dx and dI/dy with the Sobel operator:
       Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
       Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
3. **Temporal gradient** — dI/dt = frame(t+1) − frame(t).
4. **Exhaustive MSE search** — For each grid cell, try all integer (u, v) with
   |u| ≤ max_displacement and |v| ≤ max_displacement. For each candidate, compute
   the Mean Square Error of the brightness constancy residual over the 3×3 region:
       MSE(u, v) = (1/9) Σ (dI/dx * u + dI/dy * v + dI/dt)²
   The (u, v) that minimises this MSE is selected.
5. **Motion segmentation** — Compare estimated flow against predicted ego-motion to
   flag independently moving objects.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reuse convolution & kernels from the existing image_processing module.
# SOBEL_Y is defined locally because the course-required kernel
#     Sy = [[-1,-2,-1],[0,0,0],[1,2,1]]
# has the opposite sign from GY in image_processing.
# ---------------------------------------------------------------------------
from image_processing.image_processing import (
    convolution,               # 2-D convolution (same algorithm as the removed convolve2d)
    GAUSSIAN_KERNEL_3x3,       # (3×3) Gaussian blur kernel, weights sum to 16
    GX as SOBEL_X,             # Sobel kernel for dI/dx
)

# Sobel kernel for dI/dy (course-required sign convention)
SOBEL_Y = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]], dtype=np.float64
)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB (or RGBA) image to a single-channel grayscale float image
    using the standard luminance weights (ITU-R BT.601).

    Parameters
    ----------
    image : ndarray of shape (H, W, 3) or (H, W, 4)

    Returns
    -------
    gray : ndarray of shape (H, W), dtype float64, values in [0, 255]
    """
    # Use only the first three channels (drop alpha if present)
    return (0.299 * image[:, :, 0].astype(np.float64)
            + 0.587 * image[:, :, 1].astype(np.float64)
            + 0.114 * image[:, :, 2].astype(np.float64))


# ---------------------------------------------------------------------------
# Core optical flow computation
# ---------------------------------------------------------------------------


def compute_gradients(gray_prev: np.ndarray, gray_curr: np.ndarray):
    """
    Compute the three partial derivatives required by the brightness constancy
    equation.

    Spatial gradients (dI/dx, dI/dy):
        These describe how intensity changes in the x and y directions at each
        pixel. Computed by convolving the *current* frame with the Sobel kernels.

        Sx = [[-1,0,1],[-2,0,2],[-1,0,1]]   →  dI/dx
        Sy = [[-1,-2,-1],[0,0,0],[1,2,1]]    →  dI/dy

    Temporal gradient (dI/dt):
        The change in brightness at the same (x,y) location between two successive
        frames. Approximated as:  dI/dt = I(t+1) − I(t)

    Parameters
    ----------
    gray_prev : 2-D float array — the blurred grayscale image at time t
    gray_curr : 2-D float array — the blurred grayscale image at time t+1

    Returns
    -------
    Ix, Iy, It : three 2-D float arrays (same shape as input)
    """
    # Sobel filtering for spatial gradients
    Ix = convolution(gray_curr, SOBEL_X)  # dI/dx
    Iy = convolution(gray_curr, SOBEL_Y)  # dI/dy

    # Temporal gradient (pixel-wise difference between frames)
    It = gray_curr - gray_prev  # dI/dt

    return Ix, Iy, It


def compute_optical_flow_grid(
    Ix: np.ndarray,
    Iy: np.ndarray,
    It: np.ndarray,
    grid_step: int = 8,
    patch_radius: int = 1,
    max_displacement: int = 2,
):
    """
    Estimate the optical flow field on a regular grid by exhaustive MSE
    minimisation of the brightness constancy residual.

    For every grid point we extract a (2*patch_radius+1)² patch (default 3×3,
    giving 9 equations) and evaluate every integer displacement (u, v) with
    |u| ≤ max_displacement and |v| ≤ max_displacement.

    The residual for one pixel (xi, yi) in the patch is:
        r_i  =  Ix(xi, yi) · u  +  Iy(xi, yi) · v  +  It(xi, yi)

    which comes directly from the linearised brightness constancy equation:
        dI/dx · u  +  dI/dy · v  +  dI/dt  ≈  0

    The (u, v) that minimises Mean Square Error over the patch is selected:
        MSE(u, v) = (1 / N) Σ r_i²

    where N = (2*patch_radius+1)².  This is the *region assumption*: pixels
    in the neighbourhood share the same motion vector.

    Parameters
    ----------
    Ix, Iy, It    : spatial & temporal gradient images (H, W)
    grid_step     : spacing (in pixels) between sampled grid points
    patch_radius  : half-size of the neighbourhood (1 → 3×3 patch)
    max_displacement : maximum magnitude for u or v (search bound)

    Returns
    -------
    points : ndarray (N, 2) — (x, y) locations of grid points
    flows  : ndarray (N, 2) — estimated (u, v) flow at each point
    """
    h, w = Ix.shape
    pr = patch_radius
    md = max_displacement

    points_list = []
    flows_list = []

    # Build the set of candidate displacements once
    displacements = []
    for du in range(-md, md + 1):
        for dv in range(-md, md + 1):
            displacements.append((du, dv))
    displacements = np.array(displacements, dtype=np.float64)  # shape (K, 2)

    # Iterate over grid points, staying away from borders
    for y in range(pr + md, h - pr - md, grid_step):
        for x in range(pr + md, w - pr - md, grid_step):
            # Extract the 3×3 patch of gradient values
            ix_patch = Ix[y - pr: y + pr + 1, x - pr: x + pr + 1].ravel()  # (9,)
            iy_patch = Iy[y - pr: y + pr + 1, x - pr: x + pr + 1].ravel()  # (9,)
            it_patch = It[y - pr: y + pr + 1, x - pr: x + pr + 1].ravel()  # (9,)

            # -----------------------------------------------------------------
            # Exhaustive search:
            #   For each candidate (u, v), compute the residual vector
            #       r = Ix_patch * u + Iy_patch * v + It_patch
            #   and its MSE = mean(r²).
            # -----------------------------------------------------------------
            # Vectorised: residuals shape (K, 9)
            # residuals[k, :] = ix_patch * du_k + iy_patch * dv_k + it_patch
            residuals = (
                np.outer(displacements[:, 0], ix_patch)    # (K, 9)
                + np.outer(displacements[:, 1], iy_patch)  # (K, 9)
                + it_patch[np.newaxis, :]                   # broadcast (1, 9)
            )
            mse = np.mean(residuals ** 2, axis=1)  # (K,)
            best_idx = np.argmin(mse)
            best_u, best_v = displacements[best_idx]

            points_list.append((x, y))
            flows_list.append((best_u, best_v))

    points = np.array(points_list, dtype=np.float64)
    flows = np.array(flows_list, dtype=np.float64)
    return points, flows


# ---------------------------------------------------------------------------
# Motion segmentation helper
# ---------------------------------------------------------------------------


def segment_independent_motion(
    points: np.ndarray,
    flows: np.ndarray,
    ego_u: float = 0.0,
    ego_v: float = 0.0,
    threshold: float = 1.0,
):
    """
    Classify each grid point as *static background* or *independently moving*
    by comparing its flow against the dominant scene motion.

    Ego-motion is estimated automatically using the **median flow** of all
    grid points.  Because most of the scene is static background, the median
    robustly captures the camera's own motion regardless of wheel-speed
    calibration.  Points whose residual (flow minus median) exceeds the
    threshold are flagged as independently moving.

    Parameters
    ----------
    points    : (N, 2) grid locations
    flows     : (N, 2) estimated flow vectors
    ego_u, ego_v : ignored (kept for API compatibility); median is used instead
    threshold : magnitude above which a point is flagged as moving

    Returns
    -------
    labels : 1-D bool array (N,) — True for independently moving points
    """
    if len(flows) == 0:
        return np.array([], dtype=bool)

    # Robust ego-motion estimate: median of all flow vectors
    median_flow = np.median(flows, axis=0)  # (2,)

    residual = flows - median_flow
    magnitudes = np.linalg.norm(residual, axis=1)
    return magnitudes > threshold


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------


def process_frame_pair(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    grid_step: int = 8,
    patch_radius: int = 1,
    max_displacement: int = 2,
    ego_u: float = 0.0,
    ego_v: float = 0.0,
    motion_threshold: float = 1.0,
):
    # 1. Grayscale conversion
    gray_prev = rgb_to_grayscale(prev_rgb)
    gray_curr = rgb_to_grayscale(curr_rgb)

    # 2. Gaussian blur to suppress noise
    gray_prev = convolution(gray_prev, GAUSSIAN_KERNEL_3x3)
    gray_curr = convolution(gray_curr, GAUSSIAN_KERNEL_3x3)

    # 3 & 4. Compute spatial and temporal gradients
    Ix, Iy, It = compute_gradients(gray_prev, gray_curr)

    # 5. Optical flow via exhaustive MSE minimisation
    points, flows = compute_optical_flow_grid(
        Ix, Iy, It,
        grid_step=grid_step,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
    )

    # 6. Motion segmentation
    moving_mask = segment_independent_motion(
        points, flows, ego_u, ego_v, threshold=motion_threshold
    )

    return points, flows, moving_mask


def compute_flow_in_region(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    grid_step: int = 4,
    patch_radius: int = 1,
    max_displacement: int = 2,
):
    prev_crop = prev_rgb[y_min:y_max, x_min:x_max]
    curr_crop = curr_rgb[y_min:y_max, x_min:x_max]

    # Grayscale + blur
    gray_prev = rgb_to_grayscale(prev_crop)
    gray_curr = rgb_to_grayscale(curr_crop)
    gray_prev = convolution(gray_prev, GAUSSIAN_KERNEL_3x3)
    gray_curr = convolution(gray_curr, GAUSSIAN_KERNEL_3x3)
    Ix, Iy, It = compute_gradients(gray_prev, gray_curr)
    points, flows = compute_optical_flow_grid(
        Ix, Iy, It,
        grid_step=grid_step,
        patch_radius=patch_radius,
        max_displacement=max_displacement,
    )

    # Translate points back to full-image coordinates
    if len(points) > 0:
        points[:, 0] += x_min
        points[:, 1] += y_min

    return points, flows


# ---------------------------------------------------------------------------
# Lucas-Kanade single-point tracker
# ---------------------------------------------------------------------------


def lucas_kanade_track(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    points,
    window_radius: int = 7,
):
    """
    Track a list of points from the previous frame to the current frame
    using the Lucas-Kanade method.

    For each point, a (2*window_radius+1)² window is extracted.  The optical
    flow constraint  Ix·u + Iy·v + It = 0  is written for every pixel in the
    window, giving an over-determined system  A·d = b  with:

        A  = [Ix_i, Iy_i]   for each pixel i          (N×2)
        b  = [-It_i]         for each pixel i          (N×1)
        d  = [u, v]          unknown displacement      (2×1)

    The least-squares solution is  d = (AᵀA)⁻¹ Aᵀb :

        [Σ Ix²    Σ Ix·Iy] [u]   [−Σ Ix·It]
        [Σ Ix·Iy  Σ Iy²  ] [v] = [−Σ Iy·It]

    Tracking fails if the structure tensor AᵀA is near-singular (flat region
    or pure edge with no corner) or the new position leaves the image.

    Parameters
    ----------
    gray_prev     : 2-D float array — grayscale frame at time t
    gray_curr     : 2-D float array — grayscale frame at time t+1
    points        : iterable of (x, y) tuples to track
    window_radius : half-size of the tracking window (default 7 → 15×15)

    Returns
    -------
    new_points : ndarray (N, 2) — updated (x, y) positions
    status     : ndarray (N,) bool — True where tracking succeeded
    """
    # Gaussian blur to suppress noise (same as the grid-based pipeline)
    prev_blur = convolution(gray_prev, GAUSSIAN_KERNEL_3x3)
    curr_blur = convolution(gray_curr, GAUSSIAN_KERNEL_3x3)

    # Spatial and temporal gradients
    Ix = convolution(curr_blur, SOBEL_X)
    Iy = convolution(curr_blur, SOBEL_Y)
    It = curr_blur - prev_blur

    h, w = gray_curr.shape
    wr = window_radius

    new_points = []
    status = []

    for px, py in points:
        x, y = int(round(px)), int(round(py))

        # Bounds check — window must fit entirely inside the image
        if y - wr < 0 or y + wr + 1 > h or x - wr < 0 or x + wr + 1 > w:
            new_points.append((px, py))
            status.append(False)
            continue

        # Extract the window around the point
        ix_win = Ix[y - wr:y + wr + 1, x - wr:x + wr + 1].ravel()
        iy_win = Iy[y - wr:y + wr + 1, x - wr:x + wr + 1].ravel()
        it_win = It[y - wr:y + wr + 1, x - wr:x + wr + 1].ravel()

        # Structure tensor  AᵀA
        sum_ix2  = np.sum(ix_win * ix_win)
        sum_iy2  = np.sum(iy_win * iy_win)
        sum_ixiy = np.sum(ix_win * iy_win)

        # Right-hand side  Aᵀb
        sum_ixit = np.sum(ix_win * it_win)
        sum_iyit = np.sum(iy_win * it_win)

        det = sum_ix2 * sum_iy2 - sum_ixiy ** 2

        if abs(det) < 1e-4:
            # Near-singular — not enough texture to track
            new_points.append((px, py))
            status.append(False)
            continue

        # Solve the 2×2 system via explicit inverse
        inv_det = 1.0 / det
        u = inv_det * ( sum_iy2  * (-sum_ixit) - sum_ixiy * (-sum_iyit))
        v = inv_det * (-sum_ixiy * (-sum_ixit) + sum_ix2  * (-sum_iyit))

        new_x = px + u
        new_y = py + v

        # Reject if the point left the image or jumped too far
        if 0 <= new_x < w and 0 <= new_y < h and abs(u) < wr and abs(v) < wr:
            new_points.append((new_x, new_y))
            status.append(True)
        else:
            new_points.append((px, py))
            status.append(False)

    return np.array(new_points, dtype=np.float64), np.array(status, dtype=bool)
