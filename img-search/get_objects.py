import numpy as np
from pathlib import Path
import cv2
from sift import convolve_3x3, displayCv2, normalize_image
from difference import dilate_img

def grayscale_convert(img):
    gray_scales_vec = [0.2989, 0.5870, 0.1140]
    gray_scale = np.dot(img[..., :3], gray_scales_vec)
    gray_scale = gray_scale.astype(np.uint8)

    threshold = 0.15 *  np.max(gray_scale)
    binary_mask = gray_scale > threshold
    gray_scale = gray_scale * binary_mask

    return gray_scale

def get_contours(img):

    kernel = np.array([ [-0,5,  0, 0.5],
                        [-0.5,  0, 0.5],
                        [-0.5,  0, 0.5]])
    contours = convolve_3x3(img, kernel)
    contours_mask = contours > 50
    contours = contours * contours_mask

    return contours 

import cv2
import numpy as np

def remove_background_grabcut(img, output_path): #AI
    # 2. Create a mask holder (initially all zeros)
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 3. Define the background and foreground models (internal use for algorithm)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 4. Define a Rectangle (x, y, width, height)
    # We assume the object is in the center, leaving a 10px border around the edge
    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    
    # 5. Run GrabCut
    # The algorithm modifies the mask variable
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # 6. Create the final mask
    # In the mask, pixels marked 0 and 2 are background, 1 and 3 are foreground.
    # We change all background pixels to 0 and foreground to 1.
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 7. Add Alpha Channel
    # Split the image into Blue, Green, Red channels
    b, g, r = cv2.split(img)
    
    # Create the Alpha channel by scaling the mask to 0-255
    alpha = mask2 * 255
    
    # Merge the channels back together (B, G, R, A)
    rgba = cv2.merge([b, g, r, alpha])
    
    # 8. Save
    cv2.imwrite(output_path, rgba)

if __name__  == "__main__":
    in_path = Path("./cutouts")
    out_path = in_path / 'processed'
    img_paths = sorted(in_path.glob('*.jpg'))
    images = [cv2.imread(img_path) for img_path in img_paths]
    for i, img in enumerate(images):
        remove_background_grabcut(img, out_path / f"processed{i}.png")