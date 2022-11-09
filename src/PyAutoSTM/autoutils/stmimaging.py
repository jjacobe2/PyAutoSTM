''' stmimaging.py

    Module for image processing and blob detection for STM images
    to process the image and detect where CO molecules are in a given
    STM image

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 7 Nov 2022
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage

# Process image to do some denoising/contrasting
def normalize_img(img):
    ''' Normalize image for values to be between 0 and 1
    '''

    img = 1/img
    normalize_img = img.copy()

    img_max = np.amax(img)
    img_min = np.amin(img)

    normalize_img = normalize_img - (img_min) / (img_max - img_min)

    return normalize_img

# Define function what  
def contrast_img(img):
    '''
    '''
    
    #for now just use a Gaussian filter
    #img = ndimage.gaussian_filter(img, 3)
    from skimage.filters import threshold_otsu
    from scipy import fftpack

    local_thresh = threshold_otsu(img)
    img = img > local_thresh

    print(img)
    return img

# Function doing both normalizing and contrasting?
def process_img(img, disp = True):
    '''
    '''

    normalized_img = normalize_img(img)
    processed_img = contrast_img(normalized_img)

    if disp:
        fig, axes = plt.subplots(1, 3, figsize = (10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(normalized_img, cmap = 'gray')
        axes[2].imshow(processed_img, cmap = 'gray')
        plt.tight_layout()
        plt.show()

    return processed_img

# Threshold function
def threshold_img(img):
    '''
    '''

    #

    return img

# Blob detection, give pixel coordinates
def blob_detection(img):
    '''
    '''

    # 
    blobs = None

    return blobs

if __name__ == "__main__":
    pass