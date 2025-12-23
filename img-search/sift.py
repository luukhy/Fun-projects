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

@meas_time
def get_gaussian_kerenl1d(sigma: float):
    radius = int(np.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel) # to preserve img brightness

    return kernel

@meas_time
def convolve_kernel1d(img: np.ndarray, kernel:np.ndarray, axis: int, mode="edge"):
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    if axis == 0:
        pad_width = ((pad_size, pad_size), (0, 0))
    else:
        pad_width = ((0, 0), (pad_size, pad_size))

    padded_img = np.pad(img, pad_width, mode=mode)    # edge for the blurring to not produce a vinete
    windows = np.lib.stride_tricks.sliding_window_view(padded_img, kernel_size, axis=axis)
    
    return np.sum(windows * kernel, axis=-1)

@meas_time
def gaussian_blur(img:np.ndarray, sigma: float):
    kernel = get_gaussian_kerenl1d(sigma)

    blurred_horizontal  = convolve_kernel1d(img, kernel=kernel, axis=1)
    blurred_img         = convolve_kernel1d(blurred_horizontal, kernel=kernel, axis=0)

    return blurred_img


if __name__ == "__main__":
    images_paths = sorted(IMG_DIR.glob('*.jpg'))
    images = [cv2.imread(img_path) for img_path in images_paths]

    images = [images[0]]    # temporary to work on one picture for now

    for img in images:
        img = img.astype(float)
        if img.ndim == 3:
            t = time.time()
            img = np.dot(img[..., :3], GRAYSCALE_WEIGHTS)
            img = img.astype(np.uint8)
            t = time.time() - t
            print(t)

            t = time.time()
            blurred_img = gaussian_blur(img, 10.0)
            t = time.time() - t
            blurred_img = blurred_img.astype(np.uint8)
            print(t)