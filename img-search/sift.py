import numpy as np
import cv2
from pathlib import Path
import time

IMG_DIR = Path('./img') 
GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

SCALES_PER_OCTAVE = 3                                      # number of seatchable scales per octave 
DL_K_CONST = np.float32(2 ** (1 / SCALES_PER_OCTAVE))      # factor between scales
DL_SIGMA_0 = np.float32(1.6)                               # initial sigma for octaves
INITIAL_IMG_BLUR = np.float32(0.5)

DL_CONTRAST_THRESHOLD  = 0.03
DL_EDGE_THRESHOLD      = 10.0
DL_KEYPOINT_RAD_FACTOR = 3.0

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

    kernel = np.float32(1/16) * np.array((  [1, 2, 1],
                                            [1, 4, 1],
                                            [1, 2, 1] ))
    antialiased = convolve_3x3(img, kernel)
    downsampled = antialiased[::scale, ::scale]
    
    return downsampled

@meas_time
def get_gaussian_pyramid(img: np.ndarray, num_octaves:int=4) -> list:
    # because we assume that the initial image already has a blur of INITIAL_IMG_BLUR we need to calculate how much more to blur
    # to get to the desired blur of DL_SIGMA_0
    start_sigma_diff = np.sqrt(DL_SIGMA_0**2 - INITIAL_IMG_BLUR**2)
    curr_img = gaussian_blur(img, start_sigma_diff)

    gaussian_pyramid = []
    for _ in range(num_octaves):
        octave_images = [curr_img]

        for i in range(SCALES_PER_OCTAVE + 2):
            prev_sigma = (DL_K_CONST**i) * DL_SIGMA_0
            sigma_inc = prev_sigma * np.sqrt(DL_K_CONST**2 - 1)
            next_img = gaussian_blur(octave_images[-1], sigma_inc)

            octave_images.append(next_img)
        gaussian_pyramid.append(octave_images)
        base_for_next = octave_images[SCALES_PER_OCTAVE]

        curr_img = downsample(base_for_next, antialiasing=False)
    return gaussian_pyramid

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

@meas_time
def get_gradient_pyramid(gaussian_pyramid: list) -> list:
    gradient_pyramid = []
    for octave_imgs in gaussian_pyramid:
        octave_gradients = []

        for img in octave_imgs:
            dx = np.zeros_like(img, dtype=np.float32)
            dy = np.zeros_like(img, dtype=np.float32)

            # gradient calculation using central difference and slicing for optimization (approx. 20msec faster)
            dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
            dy[1:-1, :] = img[2:, :] - img[:-2, :]

            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = np.arctan2(dy, dx)
            orientation_deg = np.rad2deg(orientation)
            orientation_deg = orientation_deg % 360
            octave_gradients.append((magnitude, orientation_deg))

        gradient_pyramid.append(octave_gradients)

    return gradient_pyramid

def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    
    # avoid division by zero 
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
        
    normalized = (img - img_min) / (img_max - img_min) 
    print(np.max(normalized))
    print(np.min(normalized))
    return normalized.astype(np.uint8)

def reject_edges_hessian(axis_idxs:tuple, dog_octave: np.ndarray, edge_threshold: np.float32=DL_EDGE_THRESHOLD):
    edge_threshold_score = ((edge_threshold + 1) ** 2) / edge_threshold
    
    z_idxs, y_idxs, x_idxs = axis_idxs
    z_real = z_idxs + 1
    y_real = y_idxs + 1
    x_real = x_idxs + 1
        
    val = dog_octave[z_real, y_real, x_real]
    
    # neighbors for derivatives (Dxx, Dyy)
    val_xp = dog_octave[z_real, y_real, x_real + 1]     # right
    val_xm = dog_octave[z_real, y_real, x_real - 1]     # left
    val_yp = dog_octave[z_real, y_real + 1, x_real]     # down
    val_ym = dog_octave[z_real, y_real - 1, x_real]     # up
    
    # neighbors for cross-derivative (Dxy)
    val_xp_yp = dog_octave[z_real, y_real + 1, x_real + 1]      # bottom-Right
    val_xm_ym = dog_octave[z_real, y_real - 1, x_real - 1]      # top-Left
    val_xm_yp = dog_octave[z_real, y_real + 1, x_real - 1]      # bottom-Left
    val_xp_ym = dog_octave[z_real, y_real - 1, x_real + 1]      # top-Right

    # finite difference derivatives
    d_xx = val_xp + val_xm - 2 * val
    d_yy = val_yp + val_ym - 2 * val
    d_xy = (val_xp_yp + val_xm_ym - val_xm_yp - val_xp_ym) / 4.0

    # trace, determinant
    tr_h = d_xx + d_yy
    det_h = (d_xx * d_yy) - (d_xy ** 2)

    # Determinant > 0 (Curvature has same sign = peak/valley, not saddle)
    # Ratio Check: Tr^2 / Det < limit
    not_edge = (det_h > 0) & ((tr_h ** 2) < (edge_threshold_score * det_h))
    
    valid_z = z_real[not_edge]
    valid_y = y_real[not_edge]
    valid_x = x_real[not_edge]
    return  valid_z, valid_y, valid_x

@meas_time
def find_keypoints(dog_pyramid, contrast_threshold=DL_CONTRAST_THRESHOLD * 255):
    '''
    :param contrast_threshold:  recommended range (7.64, 25.5);
                                standard values are usually in range of (0.03; 0.10) but that is true for normalized images;
                                because I assumed standard uint8 images default threshold is set to 0.03 * 255 
    '''
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
        axis_idxs = (z_idxs, y_idxs, x_idxs)

        valid_z, valid_y, valid_x = reject_edges_hessian(axis_idxs, dog_octave)
        
        for z, y, x in zip(valid_z, valid_y, valid_x):
            keypoints.append((octave_idx, z + 1, y + 1, x + 1))

    return keypoints

def get_dominant_orientation(octave_idx: int, 
                             layer_idx: int, 
                             keypoints_by_layers: dict, 
                             orientation: np.ndarray,
                             octave_layers: list,
                             ) -> int | None:

    if (octave_idx, layer_idx) not in keypoints_by_layers:
        return None
    keypoints_in_layer = keypoints_by_layers[(octave_idx, layer_idx)]

    h, w = orientation.shape
    scale_relative = DL_SIGMA_0 * DL_K_CONST**(layer_idx)
    radius = int(DL_KEYPOINT_RAD_FACTOR * scale_relative)
    for y_pos, x_pos in keypoints_in_layer:
        y_min = y_pos - radius
        y_max = y_pos + radius
        x_min = x_pos - radius
        x_max = x_pos + radius

        gradient_grid   = octave_layers[layer_idx]
        gradient_slice  = gradient_grid[y_min:y_max, x_min:x_max] 
        

def assign_orientation(keypoints: list, gradient_pyramid: list) -> list:
    '''
    1) fore kp in keypoints:
        - get layer and position
        - get scale
        - get grid bbox (x_min, y_min, x_max, y_max)
        - get gradients in the grid
        - split the grid into quarters
        - check angles and assign them to the bins (taking the magnitude into account)
        - get the whole histogram:
            ~ check which angle is dominant -> if only one dominant: rotate by it 
                                            -> if multiple dominant (80%): rotate by every dominant creating multiples
            ~ append rotated to the oriented_keypoints
        - return oriented_keypoints
    '''
    keypoints_by_layers = {}
    for kp in keypoints:
        octave_idx, layer, y_pos, x_pos = kp
        if (octave_idx, layer) not in keypoints_by_layers:
            keypoints_by_layers[(octave_idx, layer)] = []
        keypoints_by_layers[octave_idx, layer].append((y_pos, x_pos))

    oriented_keypoints = []
    NUM_BINS = 36

    for octave_idx, octave_layers in enumerate(gradient_pyramid):
        for layer_idx, (magni, orient) in enumerate(octave_layers):
            # get dominant orientation
            orientation = get_dominant_orientation() 
            if orientation == None:
                continue



            scale_factor = 2**octave_idx
            scale_abs = scale_factor * scale_relative

    return oriented_keypoints

@meas_time
def visualize_keypoints(img, keypoints, output_path='sift_keypoints.jpg'):
    if img.ndim == 2:
        out_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        out_img = img.copy().astype(np.uint8)
        
    print(f"Drawing {len(keypoints)} keypoints...")

    for kp in keypoints:
        octave_idx, layer_idx, y, x = kp
        
        # since octave 1 is halfsize multiply coordinates by 2
        # since octave 2 is quarter size multiply by 4 and so on
        scale_multiplier = 2 ** octave_idx
        
        pt_x = int(x * scale_multiplier)
        pt_y = int(y * scale_multiplier)
        
        radius = 2 * scale_multiplier
        
        cv2.circle(out_img, (pt_x, pt_y), radius, (0, 255, 0), 1)
        
    cv2.imwrite(output_path, out_img)
    print(f"Saved visualization to {output_path}")
    return out_img

def read_images_f32(img_dir: Path=IMG_DIR, format: str='jpg'):
    path = img_dir
    format = f"*.{format}"
    images_paths = sorted(path.glob(format))
    images = [cv2.imread(str(img_path)).astype(np.float32) for img_path in images_paths]
    return images

def process_img_sift(img: np.ndarray):
    if img.ndim != 3:
        return

    img = img.astype(np.float32)
    img = np.dot(img[..., :3], GRAYSCALE_WEIGHTS)

    gaussian_pyramid =get_gaussian_pyramid(img)
    dog_pyramid      = get_dog_pyramid(gaussian_pyramid)
    gradient_pyramid = get_gradient_pyramid(gaussian_pyramid)

    keypoints = find_keypoints(dog_pyramid)
    visualize_keypoints(img, keypoints, OUTPUT_DIR / 'result_keypoints.jpg')

def sift_processing(images: list):
    for img in images:
        process_img_sift(img)

def main():
    images = read_images_f32()
    images = [images[5]]    # temporary to work on one picture for now
    sift_processing(images)

if __name__ == "__main__":
    main()