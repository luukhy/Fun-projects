import numpy as np
import cv2

def displayCv2(img):
    cv2.imshow('',img)
    cv2.waitKey(0)

def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)

    fsig = np.fft.fft2(adata)
    
    for i in range(bandlimit_index + 1, len(fsig) - bandlimit_index ):
        fsig[i] = 1000
        
    adata_filtered = np.fft.ifft2(fsig)

    return np.real(adata_filtered)

img_og = cv2.imread('G1/org.jpg')
img_ed = cv2.imread('G1/edited.jpg')

diff = img_ed - img_og
# displayCv2(diff)

gray_scales = [0.2989, 0.5870, 0.1140]
gray_diff = np.dot(diff[..., :3], gray_scales)
gray_diff = gray_diff.astype(np.uint8)

displayCv2(gray_diff)
gray_filtered = low_pass_filter(gray_diff, bandlimit=10)
gray_filtered = gray_filtered.astype(np.uint8)


gray_filtered = low_pass_filter(gray_filtered, bandlimit=10)
gray_filtered = gray_filtered.astype(np.uint8)

displayCv2(gray_filtered)
gray_filtered = low_pass_filter(gray_filtered, bandlimit=10)
gray_filtered = gray_filtered.astype(np.uint8)

displayCv2(gray_filtered)
