''' stmimaging.py

    Module for image processing and blob detection for STM images
    to process the image and detect where CO molecules are in a given
    STM image

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 18 Nov 2022
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.feature import blob_log # Import blob detection via laplacian of gaussian method

CO_EFFECTIVE_SIZE = 1.5e-9 # FOR GAUSSIAN FILTERING
CO_MIN_SIZE = 0.5e-9 # FOR BLOB DETECTION
CO_MAX_SIZE = 2e-9 # FOR BLOB DETECTION

WIDTH = 4

# Invert image so dark --> light and vice versa (Want blobs to be "on pixels" and empty space to be "off pixels")
def invert_img(img: np.ndarray):
    ''' Invert image so relative large values --> relative small values and vice versa. Do via simple inverse of data, i.e. do elementwise
    inverse of array elements

    Args:
        img (np.ndarray): image array
    '''

    return 1/img

# Normalize so all values between 0 and 1 yea?
def normalize_img(img: np.ndarray):
    ''' Normalize image for values to be between 0 and 1 (and also invert it) to prepare image for blob detection

    Args:
        img (np.ndarray): image array

    Return
        normalize_img (np.ndarray): image after inversion and normalization of data
    '''

    # Normalize by dividing by norm
    normalized_img = img / np.linalg.norm(img)
    
    return normalized_img

def denoise_image(img: np.ndarray, width: float, method: str = 'gaussian filter'):
    ''' Function to denoise image via certain methods. Default method (and only method currently implemented) is
    using scipy.ndimage.gaussian_filter function

    Args:
        img (np.ndarray): image array
        width (float): width of scan image in meters
        method (str): method to use. Default: gaussian filter

    Retrun
        denoised_img (np.ndarray): image array after denoising method
    '''

    if method == 'gaussian filter':
        
        # Figure out kernel size by using CO width in pixels as effective kernel size
        kernel_size = int(CO_EFFECTIVE_SIZE/width * 256 )
        sigma = int((kernel_size-1)/6) # apparently from this one dude on Stack Overflow, sigma should be this given kernel size.

        # Apply gaussian filter. Exponentiate to the 5th power just for the kicks, y'know? (Stop if it becomes a problem which is probs soon)
        denoised_img = ndimage.gaussian_filter(img, sigma) ** 5 

        return denoised_img

    else:
        raise ValueError('ERROR: Method chosen is invalid')

# Function to threshold/binarize image
def threshold_img(img: np.ndarray, method: str = 'otsu'):
    ''' Function to binarize image via thresholding methods in order to process image for blob detection

    Args:
        img (np.ndarray): image array
        method (str): method to use for thresholding: Default: otsu's threshold
    '''

    if method == 'otsu':
        local_thresh = threshold_otsu(img) # Find threshold using otsu's method
        bin_img = img > local_thresh # Thresholding

    else:
        raise ValueError('ERROR: Method chosen in invalid')

    return bin_img.astype(int)

# Function to do everything above in one pretty little packaged function :)
def process_img(img: np.ndarray, width: float, disp: bool = False, denoising_method: str = 'gaussian filter', thresholding_method: str = 'otsu'):
    ''' Generalized function for preprocessing image for blob detection by (1) inverting, (2) normalizing, (3) denosing, and then (4) thresholding
    the image

    Args:
        img (np.ndarray): topo or dI/dV image array
        width (float): width of image in meters
        disp (bool): boolean value of whether or not images should be displayed to see processing step by step. Default: False
        denoising_method (str): Method for denoising image. Default: gaussian filter
        thresholding_method (str): Method for thresholding/binarizing image. Default: otsu's method

    Returns:
        bin_img (np.ndarray): binarized image after all steps of processing
    '''

    inverted_img = invert_img(img) # Invert
    normalized_img = normalize_img(inverted_img) # Normalize
    denoised_img = denoise_image(normalized_img, width, denoising_method) # Denoise
    bin_img = threshold_img(denoised_img, thresholding_method) # Binarized

    if disp:
        fig, axes = plt.subplots(1, 5, figsize = (15, 5))
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(inverted_img, cmap = 'gray')
        axes[2].imshow(normalized_img, cmap = 'gray')
        axes[3].imshow(denoised_img, cmap = 'gray')
        axes[4].imshow(bin_img, cmap = 'gray')

        plt.tight_layout()
        plt.show()

    return bin_img

# Blob detection, give pixel coordinates
def blob_detection(img: np.ndarray, width: float, method: str = 'log', disp: bool = False):
    ''' Function to perform blob detection on a binarized image (intent is for image to be binarized but can theoretically be
    used without binarizing? Can be a grayscale image?)

    Args:
        img (np.ndarray): image array
        width (float): width of scan image in meters
        method (str): Method of doing blob detection. Default: Laplacian of Gaussian
        disp (bool): Display blob detection with plot. Default: False

    Returns:
        blob_coordinates (np.ndarray): array of [y, x] pixel coordinates
    '''

    # Get number of pixels along width by looking at shape of array
    pixel_width = int(img.shape[0])

    # Calculate min_sigma and max_sigma based on image sizes
    CO_pixel_width_min = int(CO_MIN_SIZE/width * pixel_width)
    CO_pixel_width_max = int(CO_MAX_SIZE/width * pixel_width)
    min_sigma = CO_pixel_width_min/(2*np.sqrt(2))
    max_sigma = CO_pixel_width_max/(2*np.sqrt(2))

    # Blob detection babyyyy
    if method == 'log':
        blobs = blob_log(img.astype(bool), min_sigma = min_sigma, max_sigma = max_sigma, overlap = 0.1)

    # If doing so, show off blob detection
    if disp:
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))
        ax.set_title('Blob Detection', fontsize = 12)

        for blob in blobs:
            y, x, area = blob
            ax.imshow(img, cmap = 'gray')
            ax.add_patch(plt.Circle((x, y), area*np.sqrt(2), color = 'r', fill = False))

        plt.tight_layout()
        plt.show()
    
    # Return only first two columns which contain the y, x coordinates
    blob_coordinates = blobs[:, :2]

    return blob_coordinates

def annihilate_square_blob(image, y_c, x_c, width = WIDTH):
    ''' Function for annihilating a square blob from an image given its coordinates
    '''

    new_image = image.copy()

    # Extra two pixels to make sure to annihilate whole blob -- DEF FIND ANOTHER WAY TO DO THIS
    width = width + 4

    # Get arrays for the x and y indices to turn on in array
    pixels_x = np.arange(x_c - int(width/2), x_c + int(width/2), 1)
    pixels_y = np.arange(y_c - int(width/2), y_c + int(width/2), 1)

    # Turn off pixels
    for pixel_y in pixels_y:
        for pixel_x in pixels_x:
            new_image[pixel_y, pixel_x] = 0

    return new_image

# Function to recursively annihilate a "blob", i.e. a group of on pixels that are neighbours. To be used for pathfinding purposes
def annihilate_blob(image, x_c, y_c):
    ''' Annihilates a group of on-pixels next to each other on an image array, where x_c and y_c can represent pixel locations of the center of the 
    blob, but it can be any location of the blob.

    Args:
        image (np.ndarray): image array
        x_c (int): pixel x coordinate/column index
        y_c (int): pixel y coordinate/row index

    Returns:
        new_image (np.ndarray): image without specificied blob
    '''

    # Create new image
    new_image = image

    # Recursive case
    if new_image[y_c, x_c] == 1:
        new_image[y_c, x_c] = 0

        # Recursive calls. Need all these if statements to not accidentally index the array with an invalid index and cause
        # Python to yell at us
        if x_c - 1 >= 0:
            annihilate_blob(new_image, y_c, x_c - 1)
        if y_c - 1 >= 0:
            annihilate_blob(new_image,  y_c - 1, x_c)
        if x_c + 1 < new_image.shape[1]:
            annihilate_blob(new_image, y_c, x_c + 1)
        if y_c + 1 < new_image.shape[0]:
            annihilate_blob(new_image, y_c + 1, x_c)
        if x_c - 1 >= 0 and y_c - 1 >= 0:
            annihilate_blob(new_image, y_c - 1, x_c - 1)
        if x_c - 1 >= 0 and y_c + 1 < new_image.shape[0]:
            annihilate_blob(new_image, y_c + 1, x_c - 1)
        if x_c + 1 < new_image.shape[1] and y_c - 1 >= 0:
            annihilate_blob(new_image, y_c - 1, x_c + 1)
        if x_c + 1 < new_image.shape[1] and y_c + 1 < new_image.shape[0]:
            annihilate_blob(new_image, y_c + 1, x_c + 1)
        
        # Base case?
        return new_image
        
    # Base case? I.e. if not 1, just return
    return new_image

# Function to delete a spherical/square blob of a certain pixel radius or whateves

# Function to enlarge blobs -- for pathfinding
def enlarge_obstacles(img: np.ndarray, pixel_size: int = 2):
    ''' Function to enlarge "obstacles", i.e. groups of on pixels, by a radius of pixel_size around all directions

    Args:
        img (np.ndarray): (binarized) image array
        pixel_size (int): radius of enlargement (that's what she said ~.o)

    Return:
        mod_img (np.ndarray): image array with obstacles enlarged
    '''
    
    mod_img = img.copy()
    height = img.shape[0]
    width = img.shape[1]

    for i in np.arange(0, height, 1):
        for j in np.arange(0, width, 1):
            if img[i, j] == 1:
                for k in np.arange(1, pixel_size + 1, 1):
                    if i - k >= 0:
                        mod_img[i - k, j] = 1
                    if j - k >= 0:
                        mod_img[i, j - k] = 1
                    if i + k < img.shape[0]:
                        mod_img[i + k, j] = 1
                    if j + k < img.shape[1]:
                        mod_img[i, j + k] = 1

    return mod_img
                
if __name__ == "__main__":
    pass