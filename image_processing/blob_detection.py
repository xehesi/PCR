import numpy as np

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
    edge_threshold=75,
):
    height, width = frame.shape[:2]
    
    if edge_magnitude.ndim > 2:
        edge_magnitude = edge_magnitude.squeeze()
    
    edges_height, edges_width = edge_magnitude.shape[:2]
    height = min(height, edges_height)
    width = min(width, edges_width)

    edge_mask = edge_magnitude[:height, :width] > edge_threshold
    
    visited = np.zeros((height, width), dtype=bool)
    all_blobs = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x] and not edge_mask[y, x]:
                target_bin = get_quantized_bin(frame[y, x])
                new_blob_pixels = []
                gradient_histogram = np.zeros(8, dtype=int)
                stack = [(x, y)]
                visited[y, x] = True
                
                while stack:
                    curr_x, curr_y = stack.pop()
                    new_blob_pixels.append((curr_x, curr_y))

                    if gradient_x is not None and gradient_y is not None:
                        gx = gradient_x[curr_y, curr_x]
                        gy = gradient_y[curr_y, curr_x]
                        if gx != 0 or gy != 0:
                            angle = np.arctan2(gy, gx) * 180 / np.pi
                            if angle < 0:
                                angle += 360
                            add_to_gradient_histogram(gradient_histogram, angle)
                    
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        neighbor_x, neighbor_y = curr_x + dx, curr_y + dy
                        if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                            if not visited[neighbor_y, neighbor_x] and not edge_mask[neighbor_y, neighbor_x]:
                                if get_quantized_bin(frame[neighbor_y, neighbor_x]) == target_bin:
                                    visited[neighbor_y, neighbor_x] = True
                                    stack.append((neighbor_x, neighbor_y))
                
                if len(new_blob_pixels) >= min_blob_size:
                    pixels_array = np.array(new_blob_pixels)
                    
                    center_x = np.mean(pixels_array[:, 0])
                    center_y = np.mean(pixels_array[:, 1])
                    
                    all_blobs.append({
                        'pixels': pixels_array,
                        'color_bin': np.array(target_bin),
                        'gradient_histogram': gradient_histogram,
                        'center': (center_x, center_y),
                        'size': len(new_blob_pixels)
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