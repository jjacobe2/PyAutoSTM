''' sim_test-2.py

    Script for running for automated assembly procedure -- similar to sim_test-1.py -- but
    with image processing and blob detection integrated, using scattering simulations in order
    to generate synthetic STM images of CO molecules on Cu(111)

    Test meant to run with Nanonis STM Simulator v5

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 18 Nov 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import time

import autoutils.astar as astar
import autocmd as cmd

from imgsim.scattered_image import create_sim_topo_image
from autoutils.stmimaging import annihilate_blob, enlarge_obstacles
from stmmap import STMMap
from tcp import Nanonis

# Settings
BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

# Main driver function to perform automation
def automation_main(stm_image: np.ndarray, desired_pos_arr: np.ndarray, centX: float, centY: float, width: float, plot_process: bool = False):
    ''' Perform automation of system specificied by desired_pos_arr in an STM image with specified image parameters and physical
    locations

    Args:   
        stm_image (np.ndarray): 2D np array representing raw STM image from scan
        desired_pos_arr (np.ndarray): Nx2 array containing locations (in units of metres) of desired locations for CO
        centX (float): x-position of center of image (m)
        centY (float): y-position of center of image (m)
        width (float): width of image (m) -> Assuming that width = height (i.e. have a square image)
        plot_process (bool): True if want to show and plot each step in process, False if not
    '''

    # Create STMMap object
    stm_map = STMMap(stm_image, centX, centY, width, width, 0)

    # Blob detection
    stm_map.process_img(stm_image, width, disp = True) # Process image
    stm_map.locate_molecules(stm_map.processed_image, width, disp = True) # Detect blobs

    # Hungarian Assignment
    stm_map.define_final_configuration(desired_pos_arr)
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

    # Initialize TCP client with STM software
    stm = Nanonis()
    stm.connect()
    stm.zctrl_onoffset(1)

    # Initialize data arrays
    V_arr = []
    I_arr = []
    z_arr = []
    t_arr = []
    pos_arr = []

    # Get starting time
    t0 = time.time()

    # Start of primary for loop
    for i in np.arange(0, desired_pos_arr.shape[0], 1):

        # Scanning mode
        cmd.scan_mode(stm, BIAS_SCAN, SETPOINT_SCAN, SPEED_MANIP)

        # Find path from current position to tip
        curr_pos = stm.folme_xyposget(wait_for_new = 0)
        curr_pos = np.array([[curr_pos[0], curr_pos[1]]])
        pixel_start = stm_map.point2pixel(curr_pos)[0]
        pixel_imol_loc = stm_map.point2pixel(np.array([stm_map.assigned_init_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array

        # Don't actually take path finding on the stm graph (i.e. the binarized image) into account for this step
        # Only need to do proper path finding when manipulating molecule
        pixel_path_arr_starttoi = astar.find_path_array(np.zeros(stm_map.raw_image.shape), 1, pixel_start, pixel_imol_loc)

        # Move to initial molecule
        V_arr, I_arr, z_arr, pos_arr, t_arr = cmd.follow_path(stm, stm_map, pixel_path_arr_starttoi, V_arr, I_arr, z_arr, pos_arr, t_arr, t0 = t0)

        # Manipulation mode
        cmd.manip_mode(stm, BIAS_MANIP, SETPOINT_MANIP, SPEED_MANIP)

        # Modify map by removing the molecule to be moved from the map in order to not consider it for path finding
        pixel_fmol_loc = stm_map.point2pixel(np.array([stm_map.assigned_final_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array
        stm_img_image_path = stm_map.processed_image.copy() # Before removing molecule from map, save image into copy for use for showing path finding
        stm_img_image = annihilate_blob(stm_map.processed_image.copy(), pixel_imol_loc[0], pixel_imol_loc[1]) # Take molecule we're working with in map/image for path finding purposes

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
        V_arr, I_arr, z_arr, pos_arr, t_arr = cmd.follow_path(stm, stm_map, pixel_path_arr_itof, V_arr, I_arr, z_arr, pos_arr, t_arr, t0 = t0)

        # "Scan"/i.e. regenerate new image
        stm_map.confirm_successful_move(i)
        molecule_locations = stm_map.unmoved_all_molecules.tolist()

        for j in range(i):
            molecule_locations = molecule_locations + [desired_pos_arr[j, :].tolist()]

        molecule_locations = np.array(molecule_locations)
        i_img = create_sim_topo_image(molecule_locations, img_width, bias_V, num_pixels, integration_points)

        plt.imshow(i_img, cmap = 'gray')
        plt.show()

        # Do image processing to see if moved successfully - Optional for now bruh
        stm_map.process_img(i_img, width, disp = True)
    
if __name__ == "__main__":

    # Create image
    img_width = 20e-9
    bias_V = 0.5
    num_pixels = 256
    integration_points = 20
    molecules_pos_arr = np.array([[7, 3], [5, 5], [9, 7], [3.5, 5], [4, -2], [-4, -4], [5, -5], [-7, -7], [-5, 6]]) * 1e-9

    image = create_sim_topo_image(molecules_pos_arr, img_width, bias_V, num_pixels, integration_points)
    desired_pos_arr = np.array([[2, -2], [2, 2], [-2, 2], [-2, -2]]) * 1e-9

    centX = 0
    centY = 0
    width = img_width

    automation_main(image, desired_pos_arr, centX, centY, width, plot_process = True)