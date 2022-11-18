''' basic_image.py

    Module for functions for generating basic images of
    blobs on an empty background. Images implemented as 2D numpy.ndarray

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 28 Oct 2022
'''

import numpy as np

# Constant pixel width of blob
WIDTH = 4

def generate_square_blob(image, y_c, x_c, width = WIDTH):
    ''' Function to generate a "square blob" on an image represented by
    a boolean 2D numpy array

    Args:
        image (np.array): image represented as binary 2d array
        x_c (int): pixel x location of center of blob 
        y_c (int): pixel y location of center of blob (note: origin is top left of image)
        width (int): desired width of blob in pixels
    '''

    new_image = image.copy()

    # Get arrays for the x and y indices to turn on in array
    pixels_x = np.arange(x_c - int(width/2), x_c + int(width/2), 1)
    pixels_y = np.arange(y_c - int(width/2), y_c + int(width/2), 1)

    # Turn on pixels
    for pixel_y in pixels_y:
        for pixel_x in pixels_x:
            new_image[pixel_y, pixel_x] = 1

    return new_image
    
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

def generate_example_image(molecule_R, N=256, M=256):
    ''' Function to generate a toy STM image

        For now, constrained to being a 256x256 image

        Args:
            molecule_R (Nx2): pixels representing locations of molecules.
            For a single molecule_r, the first entry (0th index) is y while the
            second entry (1st index) is x
    '''

    # Initialize image
    img = np.zeros((N, M))

    # Generate blobs at desired locations
    for molecule_r in molecule_R:
        img = generate_square_blob(img, molecule_r[0], molecule_r[1])

    return img

if __name__ == "__main__":
    locs = np.array([[50, 50], [75, 75], [60, 57], [49, 48]])
    img = generate_example_image(locs)

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()