import numpy as np
import cv2
from pathlib import Path
import time

IMG_DIR = Path('./img') 
GRAYSCALE_WEIGHTS = [0.299, 0.587, 0.114]

SCALES_PER_OCTAVE = 3                       # number of scales per octave in the pyramid
K_CONST = 2 ** (1 / SCALES_PER_OCTAVE)      # factor between scales
SIGMA_0 = 1.6                               # initial sigma for octaves
INITIAL_IMG_BLUR = 0.5

OUTPUT_DIR = Path('./out')

def meas_time(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()

        duration = after - before
        print(f"Function {func.__name__} took: {duration}s")
        return result

    return wrapper

@meas_time
def convolve_3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.shape != (3,3):
        print("Incorrect kernel size! returning unchanged input")
        return img 
    
    top_left    = img[0:-2, 0:-2]
    top_mid     = img[0:-2, 1:-1]
    top_right   = img[0:-2, 2:] 

    mid_left    = img[1:-1, 0:-2]
    center      = img[1:-1, 1:-1]
    mid_right   = img[1:-1, 2:]

    bot_left    = img[2:, 0:-2]
    bot_mid     = img[2:, 1:-1]
    bot_right   = img[2:, 2:]
    
    output=(    (kernel[0][0] * top_left) + (kernel[0][1] * top_mid) + (kernel[0][2] * top_right) +
                (kernel[1][0] * mid_left) + (kernel[1][1] * center)  + (kernel[1][2] * mid_right) +
                (kernel[2][0] * bot_left) + (kernel[2][1] * bot_mid) + (kernel[2][2] * bot_right)
    )

    output = np.abs(output)

    output = np.clip(output, 0, 255)

    return output

def displayCv2(img: np.ndarray):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_gaussian_kerenl1d(sigma: float) -> np.ndarray:
    RADIUS_2_SIGMA_RATIO = 3
    radius = int(np.ceil(sigma * RADIUS_2_SIGMA_RATIO))
    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)            # to preserve img brightness

    return kernel

def convolve_kernel1d_slow(img: np.ndarray, kernel:np.ndarray, axis: int, mode="edge") -> np.ndarray:
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    if axis == 0:
        pad_width = ((pad_size, pad_size), (0, 0))
    else:
        pad_width = ((0, 0), (pad_size, pad_size))

    padded_img = np.pad(img, pad_width, mode=mode)    # edge for the blurring to not produce a vinete
    windows = np.lib.stride_tricks.sliding_window_view(padded_img, kernel_size, axis=axis)
    
    return np.sum(windows * kernel, axis=-1)

def convolve_kernel1d(img: np.ndarray, kernel: np.ndarray, batch_size: int=256) -> np.ndarray:
    """
    convolves using sliding windows but processes in chunks to save memory and time
    """
    pad_size = len(kernel) // 2
    pad_width = ((0, 0), (pad_size, pad_size))
        
    padded_image = np.pad(img, pad_width, mode='edge')
    
    num_rows = img.shape[0]
    kernel_len = len(kernel)
    
    row_start = pad_width[0][0]     # to skip added vertical padding
    row_end = row_start + num_rows
    relevant_rows = padded_image[row_start:row_end, :]

    windows = np.lib.stride_tricks.sliding_window_view(relevant_rows, kernel_len, axis=1)

    return np.dot(windows, kernel)

@meas_time
def gaussian_blur(img:np.ndarray, sigma: float) -> np.ndarray:
    kernel = get_gaussian_kerenl1d(sigma)

    blurred_horizontal  = convolve_kernel1d(img, kernel=kernel)

    # rotate the og image to apply the same convolution on colums to avoid math-heavy striding
    rotated_img     = np.ascontiguousarray(blurred_horizontal.T )
    blurred_rotated = convolve_kernel1d(rotated_img, kernel=kernel)
    return blurred_rotated.T

def downsample(img: np.ndarray, scale: int=2, antialiasing=False) -> np.ndarray:
    if antialiasing == False:
        return img[::scale, ::scale]

    kernel = 1/16 * np.array((  [1, 2, 1],
                                [1, 4, 1],
                                [1, 2, 1] ))
    antialiased = convolve_3x3(img, kernel)
    downsampled = antialiased[::scale, ::scale]
    
    return downsampled

@meas_time
def get_gaussian_pyramid(img: np.ndarray, num_octaves:int=4) -> list:
    # because we assume that the initial image already has a blur of INITIAL_IMG_BLUR we need to calculate how much more to blur
    # to get to the desired blur of SIGMA_0
    start_sigma_diff = np.sqrt(SIGMA_0**2 - INITIAL_IMG_BLUR**2)
    curr_img = gaussian_blur(img, start_sigma_diff)

    pyramid = []
    for _ in range(num_octaves):
        octave_images = [curr_img]
        for i in range(SCALES_PER_OCTAVE + 2):
            prev_sigma = (K_CONST**i) * SIGMA_0
            sigma_inc = prev_sigma * np.sqrt(K_CONST**2 - 1)
            next_img = gaussian_blur(octave_images[-1], sigma_inc)

            octave_images.append(next_img)
        pyramid.append(octave_images)
        base_for_next = octave_images[SCALES_PER_OCTAVE]

        curr_img = downsample(base_for_next, antialiasing=False)
    return pyramid

@meas_time
def get_dog_pyramid(gaussian_pyramid: list) -> list:
    dog_pyramid = []
    for octave_imgs in gaussian_pyramid:
        dog_images = []
        for i in range(len(octave_imgs) - 1):
            dog = octave_imgs[i+1] - octave_imgs[i]
            dog_images.append(dog)
        dog_pyramid.append(dog_images)
    return dog_pyramid

def get_gradient_pyramid(gaussian_pyramid: list) -> list:
    pass

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    
    # avoid division by zero 
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
        
    normalized = (img - img_min) / (img_max - img_min) * 255
    return normalized.astype(np.uint8)

@meas_time
def find_keypoints_numpy(dog_pyramid, contrast_threshold=1):
    keypoints = []

    for octave_idx, dog_octave_list in enumerate(dog_pyramid):
        
        dog_octave = np.array(dog_octave_list)
        
        center_pixel = dog_octave[1:-1, 1:-1, 1:-1]
        
        is_max = np.ones(center_pixel.shape, dtype=bool)
        is_min = np.ones(center_pixel.shape, dtype=bool)
        
        # ugly but best i can do for now
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue 

                    # Map -1/0/1 to slices
                    sl_z = slice(1+dz, -1+dz) if -1+dz < 0 else slice(1+dz, None)
                    sl_y = slice(1+dy, -1+dy) if -1+dy < 0 else slice(1+dy, None)
                    sl_x = slice(1+dx, -1+dx) if -1+dx < 0 else slice(1+dx, None)

                    neighbor = dog_octave[sl_z, sl_y, sl_x]
                    
                    is_max &= (center_pixel > neighbor)
                    is_min &= (center_pixel < neighbor)

        is_strong = np.abs(center_pixel) > contrast_threshold
        
        extrema_mask = (is_max | is_min) & is_strong
        
        z_idxs, y_idxs, x_idxs = np.where(extrema_mask)
        
        for z, y, x in zip(z_idxs, y_idxs, x_idxs):
            keypoints.append((octave_idx, z + 1, y + 1, x + 1))

    return keypoints

@meas_time
def visualize_keypoints(img, keypoints, output_path='sift_keypoints.jpg'):
    '''
    temporary func
    '''
    if img.ndim == 2:
        out_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        out_img = img.copy().astype(np.uint8)
        
    print(f"Drawing {len(keypoints)} keypoints...")

    for kp in keypoints:
        octave_idx, layer_idx, y, x = kp
        
        # since octave 1 is halfsize multiply coordinates by 2
        # since octave 2 is quarter size multiply by 4
        scale_multiplier = 2 ** octave_idx
        
        pt_x = int(x * scale_multiplier)
        pt_y = int(y * scale_multiplier)
        
        radius = 2 * scale_multiplier
        
        cv2.circle(out_img, (pt_x, pt_y), radius, (0, 255, 0), 1)
        
    cv2.imwrite(output_path, out_img)
    print(f"Saved visualization to {output_path}")
    return out_img

def read_images(img_dir: Path=IMG_DIR, format: str='jpg'):
    path = img_dir
    format = f"*.{format}"
    images_paths = sorted(path.glob(format))
    images = [cv2.imread(img_path).astype(np.float32) for img_path in images_paths]
    return images

def process_img_sift(img: np.ndarray):
    img = img.astype(float)
    if img.ndim == 3:
        img = np.dot(img[..., :3], GRAYSCALE_WEIGHTS)
        # img = np.zeros((500, 500), dtype=np.float32)
        
        # # 2. Draw a white square in the middle (200x200)
        # img[150:350, 150:350] = 255.0
        cv2.imwrite(OUTPUT_DIR / 'out_1.jpg', img.astype(np.uint8))

        gaussian_pyramid = get_gaussian_pyramid(img)
        gaussian_dog     = get_dog_pyramid(gaussian_pyramid)

        cv2.imwrite(OUTPUT_DIR / 'pyramid00.jpg', gaussian_pyramid[0][0].astype(np.uint8))
        cv2.imwrite(OUTPUT_DIR / 'pyramid10.jpg', gaussian_pyramid[1][0].astype(np.uint8))
        cv2.imwrite(OUTPUT_DIR / 'dog00.jpg', normalize_image(gaussian_dog[0][0]))
        # displayCv2(normalize_image(gaussian_dog[1][4]))
        keypoints = find_keypoints_numpy(gaussian_dog)
        visualize_keypoints(img, keypoints, OUTPUT_DIR / 'result_keypoints.jpg')

def sift_processing(images: list):
    for img in images:
        process_img_sift(img)

def main():
    images = read_images()
    images = [images[5]]    # temporary to work on one picture for now
    sift_processing(images)

if __name__ == "__main__":
    main()