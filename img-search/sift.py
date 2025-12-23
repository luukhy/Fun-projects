import numpy as np
import cv2
from pathlib import Path
import time

IMG_DIR = Path('./img') 
GRAYSCALE_WEIGHTS = [0.299, 0.587, 0.114]

def meas_time(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()

        duration = after - before
        print(f"Function {func.__name__} took: {duration}s")
        return result

    return wrapper

def displayCv2(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_gaussian_kerenl1d(sigma: float):
    radius = int(np.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel) # to preserve img brightness

    return kernel

def convolve_kernel1d_slow(img: np.ndarray, kernel:np.ndarray, axis: int, mode="edge"):
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    if axis == 0:
        pad_width = ((pad_size, pad_size), (0, 0))
    else:
        pad_width = ((0, 0), (pad_size, pad_size))

    padded_img = np.pad(img, pad_width, mode=mode)    # edge for the blurring to not produce a vinete
    windows = np.lib.stride_tricks.sliding_window_view(padded_img, kernel_size, axis=axis)
    
    return np.sum(windows * kernel, axis=-1)

def convolve_kernel1d(image, kernel, batch_size=256):
    """
    convolves using sliding windows but processes in chunks to save memory and time
    """
    pad_size = len(kernel) // 2
    pad_width = ((0, 0), (pad_size, pad_size))
        
    padded_image = np.pad(image, pad_width, mode='edge')
    
    num_rows = image.shape[0]
    kernel_len = len(kernel)
    
    row_start = pad_width[0][0]     # to skip added vertical padding
    row_end = row_start + num_rows
    relevant_rows = padded_image[row_start:row_end, :]

    windows = np.lib.stride_tricks.sliding_window_view(relevant_rows, kernel_len, axis=1)

    return np.dot(windows, kernel)

@meas_time
def gaussian_blur(img:np.ndarray, sigma: float):
    kernel = get_gaussian_kerenl1d(sigma)

    blurred_horizontal  = convolve_kernel1d(img, kernel=kernel)

    # rotate the og image to apply the same convolution on colums to avoid heavy math striding
    rotated_img     = np.ascontiguousarray(blurred_horizontal.T )
    blurred_rotated = convolve_kernel1d(rotated_img, kernel=kernel)
    return blurred_rotated.T


if __name__ == "__main__":
    images_paths = sorted(IMG_DIR.glob('*.jpg'))
    images = [cv2.imread(img_path).astype(np.float32) for img_path in images_paths]

    images = [images[0]]    # temporary to work on one picture for now

    for img in images:
        img = img.astype(float)
        if img.ndim == 3:
            img = np.dot(img[..., :3], GRAYSCALE_WEIGHTS)
            cv2.imwrite('out1.jpg', img.astype(np.uint8))

            blurred_img = gaussian_blur(img, 4)
            cv2.imwrite('out2.jpg', blurred_img.astype(np.uint8))