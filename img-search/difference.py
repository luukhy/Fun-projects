import numpy as np
import cv2

def displayCv2(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_bbox_bfs(non_zero_idx: np.ndarray):
    '''
    1) choose a point from non_zero_idx
    2) start bfs from that point 
    3) when visiting remove from the non_zero_idx
    4) update min_x/y and max_x/y for bbox boundries
    5) when bfs finishes start with a not-yet-removed non_zero_index
    6) repeat until non_zero_idx empty
    '''

def convolve_3x3(in_arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.shape != (3,3):
        print("Incorrect kernel size! returning unchanged input")
        return in_arr
    # create multiple views of the input array so that there is no looping 

    img = in_arr.astype(float)

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

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

# img_og = cv2.imread('G1/org.jpg').astype(float)
# img_ed = cv2.imread('G1/edited.jpg').astype(float)

img_og = cv2.imread('G2/dublin.jpg').astype(float)
img_ed = cv2.imread('G2/dublin_edited.jpg').astype(float)

# img_og = cv2.imread('G2/london.jpg').astype(float)
# img_ed = cv2.imread('G2/london_ed.jpg').astype(float)

diff = np.absolute(img_ed - img_og)
diff = diff.astype(np.uint8)

gray_scales = [0.2989, 0.5870, 0.1140]
gray_diff = np.dot(diff[..., :3], gray_scales)
gray_diff = gray_diff.astype(np.uint8)
displayCv2(gray_diff)

threshold = 0.15 *  np.max(gray_diff)
binary_mask = gray_diff > threshold
gray_diff = gray_diff * binary_mask

hp_kernel = np.array([  [1,  1, 1],
                        [1, -8, 1],
                        [1,  1, 1]])

contours = convolve_3x3(gray_diff, hp_kernel)

non_zero = np.nonzero(contours)