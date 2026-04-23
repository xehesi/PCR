import numpy as np
import math

def get_quantized_index(pixel_rgb):
    r, g, b = pixel_rgb
    return (r // 32, g // 32, b // 32)

def is_within_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height

def get_quantized_bin(pixel_rgb):
    return tuple((pixel_rgb // 32).astype(int))

def add_to_gradient_histogram(gradient_histogram, angle):
    bin_index = int((angle + 22.5) % 360 // 45)
    gradient_histogram[bin_index] += 1


def detect_blob(
    frame,
    edge_magnitude,
    gradient_x=None,
    gradient_y=None,
    min_blob_size=10,
    max_blob_size=None,
    edge_threshold=75,
    motion_mask=None,
):
    height, width = frame.shape[:2]
    
    if edge_magnitude.ndim > 2:
        edge_magnitude = edge_magnitude.squeeze()
    
    edges_height, edges_width = edge_magnitude.shape[:2]
    height = min(height, edges_height)
    width = min(width, edges_width)

    edge_mask = edge_magnitude[:height, :width] > edge_threshold

    # When a motion_mask is provided, only seed blobs from pixels that have
    # changed between frames.  The flood fill is still allowed to grow into
    # static neighbours (so the full blob shape is recovered), but we never
    # *start* a new blob on a pixel that hasn't moved.
    has_motion_mask = motion_mask is not None

    # Pre-compute quantized colour bins for the whole image at once (avoids
    # calling get_quantized_bin per-pixel inside the flood-fill loop).
    quantized = (frame[:height, :width] // 32).astype(np.int32)
    
    visited = np.zeros((height, width), dtype=bool)
    has_gradients = gradient_x is not None and gradient_y is not None
    all_blobs = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or edge_mask[y, x]:
                continue
            # Only start a blob on a pixel that is inside the motion region
            if has_motion_mask and not motion_mask[y, x]:
                continue

            target_r = quantized[y, x, 0]
            target_g = quantized[y, x, 1]
            target_b = quantized[y, x, 2]
            new_blob_pixels = []
            gradient_histogram = np.zeros(8, dtype=int)
            stack = [(x, y)]
            visited[y, x] = True
            
            while stack:
                curr_x, curr_y = stack.pop()
                new_blob_pixels.append((curr_x, curr_y))

                if has_gradients:
                    gx = gradient_x[curr_y, curr_x]
                    gy = gradient_y[curr_y, curr_x]
                    if gx != 0 or gy != 0:
                        angle = math.atan2(gy, gx) * 57.29577951308232  # 180/pi
                        if angle < 0:
                            angle += 360.0
                        add_to_gradient_histogram(gradient_histogram, angle)
                
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_x, neighbor_y = curr_x + dx, curr_y + dy
                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        if not visited[neighbor_y, neighbor_x] and not edge_mask[neighbor_y, neighbor_x]:
                            q = quantized[neighbor_y, neighbor_x]
                            if q[0] == target_r and q[1] == target_g and q[2] == target_b:
                                visited[neighbor_y, neighbor_x] = True
                                stack.append((neighbor_x, neighbor_y))
            
            blob_size = len(new_blob_pixels)
            if blob_size < min_blob_size:
                continue
            if max_blob_size is not None and blob_size > max_blob_size:
                continue

            pixels_array = np.array(new_blob_pixels)
            
            center_x = np.mean(pixels_array[:, 0])
            center_y = np.mean(pixels_array[:, 1])
            
            all_blobs.append({
                'pixels': pixels_array,
                'color_bin': np.array([target_r, target_g, target_b]),
                'gradient_histogram': gradient_histogram,
                'center': (center_x, center_y),
                'size': blob_size
            })
                    
    return all_blobs

def group_blobs(all_blobs, distance_threshold=10):
    grouped_blobs = []
    visited = [False] * len(all_blobs)

    for i, blob in enumerate(all_blobs):
        if visited[i]:
            continue
        
        group = [blob]
        visited[i] = True
        center_x, center_y = blob['center']
        
        for j in range(i + 1, len(all_blobs)):
            if visited[j]:
                continue
            
            other_blob = all_blobs[j]
            other_center_x, other_center_y = other_blob['center']
            distance = np.sqrt((center_x - other_center_x) ** 2 + (center_y - other_center_y) ** 2)
            
            if distance <= distance_threshold:
                group.append(other_blob)
                visited[j] = True
        
        grouped_blobs.append(group)
    
    return grouped_blobs

def blobs_to_image(all_blobs, width, height):
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for blob in all_blobs:
        color_bin = blob['color_bin']
        color = (color_bin * 32 + 16).astype(np.uint8)
        
        pixels = blob['pixels']
        output_image[pixels[:, 1], pixels[:, 0]] = color
            
    return output_image


def find_ball_blob(all_blobs, width, height):
    best_blob = None
    img_center_x, img_center_y = width / 2, height / 2
    min_dist = float('inf')

    for blob in all_blobs:
        if 50 < blob['size'] < (width * height * 0.4):
            blob_c_x, blob_c_y = blob['center']
            dist = (blob_c_x - img_center_x)**2 + (blob_c_y - img_center_y)**2

            if dist < min_dist:
                min_dist = dist
                best_blob = blob

    return best_blob


# ── Motion detection helpers (Phase 4 sensor fusion) ────────────────────────


def compute_motion_mask(prev_gray, curr_gray, threshold=20):
    """
    Frame-difference motion detector — written from scratch, no cv2.
    Returns a boolean (H, W) array: True where pixel intensity changed by
    more than `threshold` between the two grayscale frames.
    """
    diff = np.abs(curr_gray.astype(np.int32) - prev_gray.astype(np.int32))
    return diff > threshold


def get_dynamic_lidar_indices(motion_mask, cam_fov_rad, lidar_angles_rad, cam_width):
    """
    Map camera-column motion to LiDAR ray indices.

    Convention (matching webot_graphSLAM.py):
      - Column 0          → left edge  → angle = +cam_fov/2
      - Column cam_width-1 → right edge → angle = -cam_fov/2
      - LiDAR angle 0     → straight ahead (positive = left)

    Returns a boolean array of length len(lidar_angles_rad):
      True  → this ray points toward a region where the camera saw motion
              → exclude from the static map
      False → stationary obstacle; include normally
    """
    if not np.any(motion_mask):
        return np.zeros(len(lidar_angles_rad), dtype=bool)

    # Columns that contain at least one moving pixel
    dynamic_cols = np.where(motion_mask.any(axis=0))[0]
    if len(dynamic_cols) == 0:
        return np.zeros(len(lidar_angles_rad), dtype=bool)

    # Column index → horizontal angle (radians, left = positive)
    denom = max(cam_width - 1, 1)
    col_angles = cam_fov_rad / 2.0 - dynamic_cols * (cam_fov_rad / denom)

    # Half the angular width of one camera pixel — used as match tolerance
    half_pixel = cam_fov_rad / (2.0 * denom)

    lidar_arr = np.asarray(lidar_angles_rad, dtype=float)

    # Vectorised: (L, D) distance matrix; mark lidar ray dynamic if its angle
    # falls within half a pixel of any dynamic camera column
    diffs = np.abs(lidar_arr[:, None] - col_angles[None, :])  # (L, D)
    return diffs.min(axis=1) <= half_pixel