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
from skimage.filters import threshold_otsu

CO_EFFECTIVE_SIZE = 1.5e-9

def invert_img(img):
    ''' Invert image so relative large values --> relative small values and vice versa. Do via simple inverse of data
    '''

    return 1/img
# Process image to do some denoising/contrasting
def normalize_img(img):
    ''' Normalize image for values to be between 0 and 1 as well as also invert it
    '''

    # Invert img so where there are scatterers have higher value than "empty space"
    normalize_img = invert_img(img)
    normalize_img = normalize_img / np.linalg.norm(normalize_img)
    
    return normalize_img

# Define function what  
def threshold_img(img, width):
    '''
    '''

    kernel_size = int( CO_EFFECTIVE_SIZE/width * 256 )

    sigma = int((kernel_size-1)/6)
    print(sigma)

    # For now to process image, just f. Exponentiate to the 5th power just for the kicks, y'know?
    img = ndimage.gaussian_filter(img, sigma) ** 5 

    plt.imshow(img)
    plt.show()
    local_thresh = threshold_otsu(img)
    img = img > local_thresh

    return img

# Function doing both normalizing and contrasting?
def process_img(img, width, disp = True):
    '''
    '''

    normalized_img = normalize_img(img)
    processed_img = threshold_img(normalized_img, width)

    if disp:
        fig, axes = plt.subplots(1, 3, figsize = (10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(normalized_img, cmap = 'gray')
        axes[2].imshow(processed_img, cmap = 'gray')
        plt.tight_layout()
        plt.show()

    return processed_img

# Blob detection, give pixel coordinates
def blob_detection(img):
    '''
    '''

    blobs = None

    return blobs

if __name__ == "__main__":
    pass