''' Cs_test-1.py

    Simple complete automated assembly procedure for testing PyAutoSTM for manipulating
    Cs atoms on CsVSb

    Juwan Jeremy Jacobe
    University of Notre Dame    

    Last Modified: 30 Dec 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import time

import autoutils.astar as astar
import autocmd as cmd

from imgsim.scattered_image import create_sim_topo_image
from autoutils.stmimaging import annihilate_blob, enlarge_obstacles
from autoutils.planner import planner, rejoin_columns
from stmmap import STMMap
from tcp import Nanonis

# Settings
BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

# For generating desired pos array
def generate_square_arr(centX: float, centY: float, spacing: float, n: int) -> np.ndarray:
    ''' Generate position array for n x n Cs square, centered on (centX, centY)

    Args:
        centX (float): x-position of center of square
        centY (float): y-position of center of square
        spacing (float): spacing between Cs atoms
        n (float): number of Cs atoms per side

    Out:
        square_arr (np.ndarray): position array for Cs atoms in the square
    '''

    # Initiate
    square_arr = []

    # Generate top-line and bottom line
    for i in np.arange(-int(n/2), int(n/2)+1, 1):
        square_arr = square_arr + [[centX + i * spacing, centY + int(n/2) * spacing]] + [[centX + i * spacing, centY - int(n/2) * spacing]]

    # Generate side lines
    for i in np.arange(-(int(n/2)-1), int(n/2), 1):
        square_arr = square_arr + [[centX + int(n/2) * spacing, centY + i * spacing]] + [[centX - int(n/2) * spacing, centY + i * spacing]]

    # Convert nested list to np.ndarray
    square_arr = np.array(square_arr)

    return square_arr

# Main driver function to perform automation
def automation_main(desired_pos_arr: np.ndarray, centX: float, centY: float, width: float, plot_process: bool = False):
    ''' Perform automation of system specificied by desired_pos_arr in an STM image with specified image parameters and physical
    locations

    Args:   
        desired_pos_arr (np.ndarray): Nx2 array containing locations (in units of metres) of desired locations for CO
        centX (float): x-position of center of image (m)
        centY (float): y-position of center of image (m)
        width (float): width of image (m) -> Assuming that width = height (i.e. have a square image)
        plot_process (bool): True if want to show and plot each step in process, False if not
    '''

    # Initialize TCP client
    stm = Nanonis()
    stm.connect()

    # Turn on Z-Ctrl
    stm.zctrl_onoffset(1)

    # Perform scan
    stm_image = cmd.perform_scan(stm, centX, centY, width, angle = 0)

    # Create map object from scan
    stm_map = STMMap(stm_image, centX, centY, width, width, angle = 0)

    # Blob detection
    stm_map.process_img(stm_image, width, invert = False, disp = True) # Process image
    stm_map.locate_molecules(stm_map.processed_image, width, disp = True) # Blob detector proper

    # Order assembly order of manipulation
    sorted_columns = planner(desired_pos_arr, dx = 1e-9)
    srtd_pos_arr = rejoin_columns(sorted_columns)

    # Hungarian assignment
    stm_map.define_final_configuration(srtd_pos_arr)
    stm_map.assign_molecules()

    # Show hungarian assignment result
    if plot_process:

        fig = plt.figure()
        axes = plt.axes()

        # Visualize physical coordinates of molecules + coordinates of goal configuration
        axes.set_title('Physical representation')
        axes.scatter(stm_map.all_molecules[:, 0], stm_map.all_molecules[:, 1])
        axes.scatter(stm_map.assigned_final_config[:,0], stm_map.assigned_final_config[:, 1])

        for i in np.arange(0, desired_pos_arr.shape[0], 1):
            axes.plot([stm_map.assigned_init_config[i, 0], stm_map.assigned_final_config[i, 0]], 
                    [stm_map.assigned_init_config[i, 1], stm_map.assigned_final_config[i, 1]], color = "black")

        axes.set_xlim(centX - width/2, centY + width/2)
        axes.set_ylim(centY - width/2, centY + width/2)
        axes.set_aspect('equal')

        plt.show()

    # Primary loop
    for i in np.arange(0, srtd_pos_arr.shape[0], 1):

        # Scanning mode
        cmd.scan_mode(stm, BIAS_SCAN, SETPOINT_SCAN, SPEED_MANIP)
    
        # Move tip to location of molecule
        stm.folme_xyposset(stm_map.assigned_init_config[i, 0], stm_map.assigned_init_config[i, 1], wait_end_of_move = 1)

        # Manipulation mode
        cmd.manip_mode(stm, BIAS_MANIP, SETPOINT_MANIP, SPEED_MANIP)

        # Modify map by removing the molecule to be moved from the map in order to not consider it for path finding
        pixel_imol_loc = stm_map.point2pixel(np.array([stm_map.assigned_init_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array
        pixel_fmol_loc = stm_map.point2pixel(np.array([stm_map.assigned_final_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array
        stm_img_image_path = stm_map.processed_image.copy() # Before removing molecule from map, save image into copy for use for showing path finding
        stm_img_image = annihilate_blob(stm_map.processed_image.copy(), pixel_imol_loc[0], pixel_imol_loc[1]) # Take away molecule we're working with in map/image for path finding purposes   

        # Create a separate variable to store the stm binarized image and copy it
        stm_img_image_map = stm_img_image.copy()
        stm_img_image_map = enlarge_obstacles(stm_img_image_map)

        # Find path from molecule to final location it will be placed
        pixel_path_arr_itof = astar.find_path_array(stm_img_image_map, 1, pixel_imol_loc, pixel_fmol_loc)
     
        # If desired, show path
        if plot_process:

            # Activate pixels of path
            for coord in pixel_path_arr_itof:
                stm_img_image_path[int(coord[0]), int(coord[1])] = 1

            # Redraw molecule that was erased and then plot
            fig, axes = plt.subplots(1, 1)
            axes.imshow(stm_img_image_path, cmap = 'gray_r')
            axes.set_title(f'Path from i to f from manipulation {i}')
            plt.show()
        
        # Slowly move tip to final
        cmd.follow_path(stm, stm_map, pixel_path_arr_itof, sampling = False)

        # Scan new image to use as new map
        new_img = cmd.perform_scan(stm, centX, centY, width, 0)
        stm_map.process_img(new_img, width)

        plt.title(f'STM Image after Assembly Step {i + 1}')
        plt.imshow(stm_map.processed_image, cmap = 'gray')
if __name__ == "__main__":
    
    # Funsies header for the terminal
    print('--------------------------------------------------')
    print('|      Basic Assembly of Cs atoms on CsVSb       |')
    print('--------------------------------------------------')

    # Get desired parameters for scan frame
    centX = float(input('Enter desired x-coordinate of center of scan (m): '))
    centY = float(input('Enter desired y-coordinate of center of scan (m): '))
    width = float(input('Enter desired width of scan frame (m): '))

    # Test out n x n square generator
    n = 3
    spacing = 1e-9
    square_arr = generate_square_arr(centX, centY, spacing, 3)

    plt.scatter(square_arr[:, 0], square_arr[:, 1])
    plt.show()
