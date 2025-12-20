import numpy as np
import cv2
from collections import deque

def displayCv2(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bbox_bfs(origin: tuple, non_zero_set: set, max_iter):
    '''
    BFS algorithm that updates the boundingboxes limits. 
    The non_zer_set serves also as a UNvisited list, by removing its elemets when adding them to the bfs_q
    '''
    bfs_q = deque([])
    bfs_q.append(origin)
    non_zero_set.remove(origin)

    min_y, min_x = origin
    max_y, max_x = origin

    safety_iter = 0
    while(len(bfs_q) != 0):
        safety_iter += 1
        if safety_iter >= max_iter:
            print(f"ERROR: Exceeded maximum number of iterations ({max_iter})")
            break

        curr_y, curr_x = bfs_q.popleft()

        # TODO: make it less ugly 
        if curr_x > max_x:
            max_x = curr_x
        if curr_x < min_x:
            min_x = curr_x
        if curr_y > max_y:
            max_y = curr_y
        if curr_y < min_y:
            min_y = curr_y
        
        top     = ( curr_y - 1, curr_x      ) 
        bot     = ( curr_y + 1, curr_x      ) 
        left    = ( curr_y    , curr_x - 1  )
        right   = ( curr_y    , curr_x + 1  )

        for neighbour in [top, bot, left, right]:
            if neighbour not in non_zero_set: # ensures it is not visited and  
                continue
            bfs_q.append(neighbour)
            non_zero_set.remove(neighbour)

    return [[min_y, max_y], [min_x, max_x]]

def get_bbox_bfs(non_zero_set: set, min_blob_size = 0, max_iter = 1000000):
    '''
    1) choose a point from non_zero_idx
    2) start bfs from that point 
    3) when visiting remove from the non_zero_idx
    4) update min_x/y and max_x/y for bbox boundries
    5) when bfs finishes start with a not-yet-removed non_zero_index
    6) repeat until non_zero_idx empty
    '''
    bbox_limits = []

    while non_zero:
        origin = next(iter(non_zero_set))       # acces first element of non_zero_set
        bfs_bbox_res = bbox_bfs(origin, non_zero_set, max_iter)
        bbox_limits.append(bbox_bfs(bfs_bbox_res))

    return bbox_limits


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

# img_og = cv2.imread('G3/london.jpg').astype(float)
# img_ed = cv2.imread('G3/london_ed.jpg').astype(float)

diff = np.absolute(img_ed - img_og)
diff = diff.astype(np.uint8)

gray_scales = [0.2989, 0.5870, 0.1140]
gray_diff = np.dot(diff[..., :3], gray_scales)
gray_diff = gray_diff.astype(np.uint8)

threshold = 0.15 *  np.max(gray_diff)
binary_mask = gray_diff > threshold
gray_diff = gray_diff * binary_mask

hp_kernel = np.array([  [1,  1, 1],
                        [1, -10, 1],
                        [1,  1, 1]])

contours = convolve_3x3(gray_diff, hp_kernel)

contours_mask = contours > 50
contours = contours * contours_mask

blobing_kernel = np.array([    [1,  1, 1],
                            [1,  1, 1],
                            [1,  1, 1]])
blobs = contours
blobing_intensity = 3
for i in range(0, blobing_intensity):
    blobs = convolve_3x3(blobs, blobing_kernel)
displayCv2(blobs)

non_zero_indecies = np.nonzero(blobs)
non_zero_set = set(zip(non_zero_indecies[0], non_zero_indecies[1]))

bbox_limits = get_bbox_bfs(non_zero_set=non_zero_set)

test_copy = blobs.copy()

for bbox in bbox_limits:
    y_limit = bbox[0]