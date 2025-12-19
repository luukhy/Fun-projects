import numpy as np
import cv2
import matplotlib.pyplot as plt

TILE_SIZE = 256

def displayCv2(img):
    cv2.imshow('', img)
    cv2.waitKey(0)

def generate_tiles(padded_image, tile_size=256, stride=128):
    h_pad, w_pad = padded_image.shape[:2]
    
    stop_y = h_pad - tile_size + 1
    stop_x = w_pad - tile_size + 1
    
    for y in range(0, stop_y, stride):
        for x in range(0, stop_x, stride):
            yield padded_image[y : y + tile_size, x : x + tile_size], x, y

if __name__ == "__main__":

    # read the image
    img_ref_raw = cv2.imread('img/tiger-head-64.webp', cv2.IMREAD_UNCHANGED)
    print(type(img_ref_raw))
    img_ref_h = img_ref_raw.shape[0]
    img_ref_w = img_ref_raw.shape[1]
    img_rgb = img_ref_raw[:, :, :3]
    alpha_channel = img_ref_raw[:, :, 3]

    # calculate and apply mask
    mask = alpha_channel.astype(float) / 255.0
    mask_rgb = mask[:,:, None]
    img_float = img_rgb.astype(float) * mask_rgb
    img_ref = img_float.astype(np.uint8)

    # cv2.imshow('', img_ref)
    # cv2.waitKey(0)

    # add padding
    h_pad = ( TILE_SIZE - img_ref_h) 
    w_pad = ( TILE_SIZE - img_ref_w) 
    img_ref_pad = np.pad(img_ref, 
                         pad_width=((0, h_pad), 
                        (0, w_pad), (0, 0)), 
                        mode='constant', 
                        constant_values=0)

    displayCv2(img_ref_pad)

    # read the scene 
    img_scene = cv2.imread('img/panorama_tiger.jpg')
    img_scene_h = img_scene.shape[0]
    img_scene_w = img_scene.shape[1]

    # pad the scene
    h_pad = TILE_SIZE - (img_scene_h % TILE_SIZE)
    w_pad = TILE_SIZE - (img_scene_w % TILE_SIZE)

    img_scene_pad = np.pad(img_scene, 
                         pad_width=((0, h_pad), 
                        (0, w_pad), (0, 0)), 
                        mode='constant', 
                        constant_values=0)

    displayCv2(img_scene_pad)

    ref_fft = np.fft.fft2(img_ref_pad)
        
    stride = int(img_ref_w / 2)
    for tile in generate_tiles(img_scene_pad, TILE_SIZE, stride=stride):
        tile_fft = np.fft.fft2()
        corel = tile_fft * np.conj(ref_fft)

    # displayCv2(img_scene_pad)
