import numpy as np
from PIL import Image

GAUSSIAN_KERNEL_3x3 = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=np.float32) / 16.0
GX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
GY = np.array([[1, 2, 1],
               [0, 0, 0],
                [-1, -2, -1]], dtype=np.float32)


def convolution(image, kernel):
    image = np.array(image)
    k_h, k_w = kernel.shape
    h, w = image.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros_like(image, dtype=np.float32)
    for i in range(k_h):
        for j in range(k_w):
            output += padded_image[i:i + h, j:j + w] * kernel[i, j]

    return output


def resize_image(image, new_w, new_h):
    image = np.array(image)
    old_h, old_w = image.shape[:2]

    if image.ndim == 3:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            src_i = i * old_h // new_h
            src_j = j * old_w // new_w
            output[i, j] = image[src_i, src_j]

    return output



def edge_detection_with_gradients(image, gx, gy):
    grad_x = convolution(image, gx)
    grad_y = convolution(image, gy)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude.astype(np.uint8), grad_x, grad_y


def edge_detection_binary(image, gx, gy):
    grad_x = convolution(image, gx)
    grad_y = convolution(image, gy)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    threshold = 75
    magnitude = (magnitude > threshold) * 255

    return magnitude.astype(np.uint8)


def gaussian_blur(image, kernel, time=10):
    img_array = np.array(image)
    brightness_img = img_array[:, :, 2] if img_array.ndim == 3 else img_array
    blurred = convolution(brightness_img, kernel)
    for _ in range(time - 1):
        blurred = convolution(blurred, kernel)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
