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
import time

import utils.astar as astar

from stmmap import STMMap
from tcp import Nanonis

BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

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

def follow_path(tcp_client : Nanonis, stm_map : STMMap, path_arr : list, V_arr : list, I_arr : list, 
    z_arr : list, pos_arr : list, t_arr : list, t0: float, sampling = True):

    # Convert path arr from pixel --> x,y array
    phys_path_arr = stm_map.pixel2point(path_arr)

    # Go through each node in path and move tip to those nodes
    for point in phys_path_arr:
        tcp_client.folme_xyposset(point[0], point[1], wait_end_of_move = 1)
        if sampling:
            V_arr = V_arr + [tcp_client.bias_get()]
            I_arr = I_arr + [tcp_client.current_get()]
            z_arr = z_arr + [tcp_client.zctrl_zposget()]
            t_arr = t_arr + [time.time() - t0]
            x, y = tcp_client.folme_xyposget(wait_for_new = 0)
            pos_arr = pos_arr + [[x, y]]

    return V_arr, I_arr, z_arr, pos_arr, t_arr

def automation_main(molecule_R, final_R, centX, centY, width, height):
    ''' Run the process of automation for given test described in header docstring
    '''

    # Generate image
    test_img = generate_example_image(molecule_R)

    # Initialize STMMap object and tcp client
    stm_img = STMMap(image_data = test_img, centX = centX, centY = centY, width = width, height = height, angle = 0)
    stm = Nanonis()
    stm.connect()

    # Define desired final configuration and do assignment
    stm_img.define_final_configuration(stm_img.pixel2point(final_R))
    stm_img.all_molecules = stm_img.pixel2point(molecule_R)
    stm_img.assign_molecules()

    # Initialize settings of scan
    stm.bias_set(BIAS_SCAN)
    stm.folme_speedset(10e-9, custom_speed_mod = 0) # Use scanning speed
    stm.zctrl_onoffset(1) # Turn on

    # Initialize data arrays
    V_arr = []
    I_arr = []
    z_arr = []
    t_arr = []
    pos_arr = []

    t0 = time.time()

    fig, axes = plt.subplots(1, 2, figsize = (10, 5))

    # Visualize image
    axes[0].set_title('Binary image representation')
    axes[0].imshow(stm_img.raw_image)

    # Visualize physical coordinates of molecules + coordinates of goal configuration
    axes[1].set_title('Physical representation')
    axes[1].scatter(stm_img.all_molecules[:, 0], stm_img.all_molecules[:, 1])
    axes[1].scatter(stm_img.assigned_final_config[:,0], stm_img.assigned_final_config[:, 1])
    axes[1].set_xlim(centX - width/2, centY + width/2)
    axes[1].set_ylim(centY - height/2, centY + height/2)
    plt.show()

    # Start of primary loop for controlling
    for i in np.arange(stm_img.assigned_init_config.shape[0]):
        
        # Right now current astar implementation is finicky
        # stm_img_image = np.zeros(stm_img.raw_image.shape)
        stm_img_image = stm_img.raw_image

        # Show image before manipulation
        fig, axes = plt.subplots(1, 1)
        axes.imshow(stm_img_image, cmap = 'gray_r')
        axes.set_title(f'Image before manipulation {i}')
        plt.show()

        # Find path from current position to molecule to be moved
        curr_pos = stm.folme_xyposget(wait_for_new = 1)
        curr_pos = np.array([[curr_pos[0], curr_pos[1]]])
        pixel_start = stm_img.point2pixel(curr_pos)
        pixel_imol_loc = stm_img.point2pixel(np.array([stm_img.assigned_init_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array

        # Don't actually take path finding into account for this step -- only needs to be taken to account when manipulating molecule
        pixel_path_arr_starttoi = astar.find_path_array(np.zeros(stm_img.raw_image.shape), 1, pixel_start, pixel_imol_loc)

        # Move to initial molecule
        V_arr, I_arr, z_arr, pos_arr, t_arr = follow_path(stm, stm_img, pixel_path_arr_starttoi, V_arr, I_arr, z_arr, pos_arr, t_arr, t0 = t0)

        # Find path from initial to final
        pixel_fmol_loc = stm_img.point2pixel(np.array([stm_img.assigned_final_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array
        stm_img_image = annihilate_square_blob(stm_img_image, pixel_imol_loc[0], pixel_imol_loc[1]) # Take molecule we're working with in map/image for path finding purposes
        pixel_path_arr_itof = astar.find_path_array(stm_img_image, 1, pixel_imol_loc, pixel_fmol_loc)


        # Show path to move it
        stm_img_image_copy = stm_img_image.copy()
        for coord in pixel_path_arr_itof:
            stm_img_image_copy[coord[0], coord[1]] = 1
        axes.imshow(stm_img_image_copy, cmap = 'gray_r')
        axes.set_title(f'Path from i to f from manipulation {i}')
        plt.show()

        # Update map/image now that molecule has been move
        stm_img_image = generate_square_blob(stm_img_image, pixel_fmol_loc[0], pixel_fmol_loc[1])

        # Change settings to manipulation mode
        stm.bias_set(BIAS_MANIP)
        stm.zctrl_setpntset(SETPOINT_MANIP)
        stm.folme_speedset(SPEED_MANIP, custom_speed_mod = 1)

        # Move along path from intiial to final
        V_arr, I_arr, z_arr, pos_arr, t_arr = follow_path(stm, stm_img, pixel_path_arr_itof, V_arr, I_arr, z_arr, pos_arr, t_arr, t0 = t0)

        # Set back to scanning mode
        stm.bias_set(BIAS_SCAN)
        stm.folme_speedset(10e-9, custom_speed_mod = 0) # Use scanning speed

        # Show image after manipulation
        axes.imshow(stm_img_image, cmap = 'gray_r')
        axes.set_title(f'Image after manipulation {i}')
        plt.show()

    return stm_img

if __name__ == "__main__":

    molecule_R = np.array([[10, 20], [100, 200], [200, 200], [50, 190]])
    final_R = np.array([[120, 120], [140, 120], [130, 110], [130, 130]])
    centX = 0
    centY = 0
    width = 20e-9
    height = 20e-9

    stm_img = automation_main(molecule_R, final_R, centX, centY, width, height)

    '''
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))

    # Visualize image
    axes[0].set_title('Binary image representation')
    axes[0].imshow(stm_img.raw_image)

    # Visualize physical coordinates of molecules + coordinates of goal configuration
    axes[1].set_title('Physical representation')
    axes[1].scatter(stm_img.all_molecules[:, 0], stm_img.all_molecules[:, 1])
    axes[1].scatter(stm_img.assigned_final_config[:,0], stm_img.assigned_final_config[:, 1])
    axes[1].set_xlim(centX - width/2, centY + width/2)
    axes[1].set_ylim(centY - height/2, centY + height/2)

    plt.show()
    '''
    