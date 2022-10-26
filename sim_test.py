''' sim_test.py

    Script running tcp control test meant to run on simulator to show tcp command functionality + path finding functionality

    Test meant to run on Nanonis STM Simulator v5, dealing with an idealized "scan", with 6 molecules randomly placed. Goal of manipulating
    molecules into a hexagonal pattern

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 24 Oct 2022
'''

import numpy as np
import matplotlib.pyplot as plt

import utils.astar as astar

from stmmap import STMMap
from tcp import Nanonis

def generate_square_blob(image, y_c, x_c, width = 6):
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

def automation_main(molecule_R, final_R, centX, centY, width, height):
    ''' Run the process of automation for given test described in header docstring
    '''

    # Generate image
    test_img = generate_example_image(molecule_R)

    # Initialize STMMap object and tcp client
    stm_img = STMMap(image_data = test_img, centX = centX, centY = centY, width = width, height = height)
    stm = Nanonis()
    stm.connect()

    # Define desired final configuration and do assignment
    stm_img.define_final_configuration(final_R)
    stm_img.assign_molecules()

if __name__ == "__main__":

    molecule_R = np.array([[30, 25], [200, 180]])
    img = generate_example_image(molecule_R)
    plt.imshow(img, cmap = 'gray_r')
    plt.show()

    # automation_main()