''' sim_test.py

    Script running tcp control test meant to run on simulator to show tcp command functionality + path finding functionality

    Test meant to run on Nanonis STM Simulator v5, dealing with an idealized "scan", with 6 molecules randomly placed. Goal of manipulating
    molecules into a simple pattern

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 24 Oct 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import time

import autoutils.astar as astar

from stmmap import STMMap
from tcp import Nanonis
from imgsim.basic_image import *

# Settings
BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

def follow_path(tcp_client : Nanonis, stm_map : STMMap, path_arr : list, V_arr : list, I_arr : list, 
    z_arr : list, pos_arr : list, t_arr : list, t0: float, sampling = True):
    ''' Function for handling tcp commands to follow a certain path

    Args:
        tcp_client (Nanonis): the TCP client connected to the Nanonis SPM Controller software
        stm_map (STMMap): the image/map the tip is currently working with
        path_arr (list): nested list where each element is a list of length 2 representing the x, y variable of each node in the path
        V_arr (list): list of bias V to be measured
        I_arr (list): list of current I to be measured
        z_arr (list): list of z position Z to be measured
        pos_arr (list): list of plane position x, y to be measured
        t_arr (list): list of recorded time t to be measured
        t0 (float): reference starting time of whole automation process
        sampling (bool): 1 = want to add another observation to V_arr, I_arr, etc. 0 = don't add
    '''

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

def automation_main(molecule_R, final_R, centX, centY, width, height, obstacles, plot_img = False):
    ''' Run the process of automation for given test described in header docstring

    Args:
        molecule_R (np.ndarray): array of 2D positions of molecules, in pixel coordinates (y, x)
        final_R (np.ndarray): array of 2D positions of desired final configuration, in pixel coordinates (y, x)
        centX (float): x coordinate of center of scan frame (m)
        centY (float): y coordinate of center of scan frame (m)
        width (float): width of scan frame (m)
        height (float): height of scan frame (m)
        obstacles (np.ndarray): array of 2D positions of center of square blob ostacles, in pixel coordinates (y, x)

    Returns:
        stm_img (STMMap): the STMMap object representing the initial configuration + scan frame configuration
        V_arr (np.array): array of measured voltage biases vs time (V)
        I_arr (np.array): array of measured tunneling current vs time (A)
        z_arr (np.array): array of z position of tip vs time (m)
        pos_arr (np.array): array of measured x, y tip position vs time (m)
        t_arr (np.array): array of times associated with measurents (where t[i], for example, is time when z_arr[i] was measured) (s)
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
    stm.zctrl_setpntset(SETPOINT_SCAN)
    stm.folme_speedset(10e-9, custom_speed_mod = 0) # Use scanning speed
    stm.zctrl_onoffset(1) # Turn on

    # Initialize data arrays
    V_arr = []
    I_arr = []
    z_arr = []
    t_arr = []
    pos_arr = []

    # Get starting time
    t0 = time.time()

    # Plot representative binary image if argument is true
    if plot_img:
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
    
    # Right now current astar implementation is finicky
    stm_img_image = stm_img.raw_image
    
    # Create square blob obstacles
    for obstacle in obstacles:
        stm_img_image = generate_square_blob(stm_img_image, obstacle[0], obstacle[1])
        
    # Start of primary loop for controlling
    for i in np.arange(stm_img.assigned_init_config.shape[0]):

        # Show image before manipulation
        if plot_img:
            fig, axes = plt.subplots(1, 1)
            axes.imshow(stm_img_image, cmap = 'gray_r')
            axes.set_title(f'Image before manipulation {i}')
            plt.show()

        # Find path from current position to molecule to be moved
        curr_pos = stm.folme_xyposget(wait_for_new = 1)
        curr_pos = np.array([[curr_pos[0], curr_pos[1]]])
        pixel_start = stm_img.point2pixel(curr_pos)[0]
        pixel_imol_loc = stm_img.point2pixel(np.array([stm_img.assigned_init_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array

        # Don't actually take path finding on the stm graph (i.e. the binarized image) into account for this step
        # Only need to do proper path finding when manipulating molecule
        pixel_path_arr_starttoi = astar.find_path_array(np.zeros(stm_img.raw_image.shape), 1, pixel_start, pixel_imol_loc)

        # Move to initial molecule
        V_arr, I_arr, z_arr, pos_arr, t_arr = follow_path(stm, stm_img, pixel_path_arr_starttoi, V_arr, I_arr, z_arr, pos_arr, t_arr, t0 = t0)

        # Modify map by removing the molecule to be moved from the map in order to not consider it for path finding
        pixel_fmol_loc = stm_img.point2pixel(np.array([stm_img.assigned_final_config[i, :]]))[0] # Take 0th element as we only want a 2 vector but gives us 1 x 2 array
        stm_img_image = annihilate_square_blob(stm_img_image, pixel_imol_loc[0], pixel_imol_loc[1]) # Take molecule we're working with in map/image for path finding purposes
        
        # Create a separate variable to store the stm binarized image and copy it
        stm_img_image_map = stm_img_image.copy()

        # Enlarge the blobs 
        for py in range(stm_img_image.shape[1]):
            for px in range(stm_img_image.shape[0]):
                if stm_img_image[py, px] == 1:
                    stm_img_image_map[py - 1, px] = 1
                    stm_img_image_map[py + 1, px] = 1
                    stm_img_image_map[py, px - 1] = 1
                    stm_img_image_map[py, px + 1] = 1
                    stm_img_image_map[py - 1, px - 1] = 1
                    stm_img_image_map[py + 1, px - 1] = 1
                    stm_img_image_map[py + 1, px - 1] = 1
                    stm_img_image_map[py - 1, px + 1] = 1
                    
        # Find path from molecule to final location it will be placed
        pixel_path_arr_itof = astar.find_path_array(stm_img_image_map, 1, pixel_imol_loc, pixel_fmol_loc)

        
        # If plotting, show path
        if plot_img:

            # Create a new copy of stm map to draw path on
            stm_img_image_path = stm_img_image.copy()    

            # Activate pixels of path
            for coord in pixel_path_arr_itof:
                stm_img_image_path[int(coord[0]), int(coord[1])] = 1

            # Redraw molecule that was erased and then plot
            stm_img_image_path = generate_square_blob(stm_img_image_path, int(pixel_path_arr_itof[0, 0]), int(pixel_path_arr_itof[0, 1]))
            fig, axes = plt.subplots(1, 1)
            axes.imshow(stm_img_image_path, cmap = 'gray_r')
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
        stm.zctrl_setpntset(SETPOINT_SCAN)
        stm.folme_speedset(10e-9, custom_speed_mod = 0) # Use scanning speed

        # Show image after manipulation
        if plot_img:
            fig, axes = plt.subplots(1, 1)
            axes.imshow(stm_img_image, cmap = 'gray_r')
            axes.set_title(f'Image after manipulation {i}')
            plt.show()

    # Convert observed measurements to numpy arrays
    V_arr = np.array(V_arr)
    I_arr = np.array(I_arr)
    z_arr = np.array(z_arr)
    pos_arr = np.array(pos_arr)
    t_arr = np.array(t_arr)

    return stm_img, V_arr, I_arr, z_arr, pos_arr, t_arr

if __name__ == "__main__":

    # Create array of obstacles
    obstacles1 = [[20, 40], [40, 40], [80, 40], [60, 40], [100, 40], [120, 40], [140, 40], [160, 40], [180, 40]]
    obstacles2 = [[10, 60], [30, 60], [70, 60], [50, 60], [90, 60], [110, 60], [130, 60], [150, 60], [170, 60]]
    obstacles3 = [[20, 140], [40, 140], [80, 140], [60, 140], [100, 140], [120, 140], [140, 140], [160, 140], [180, 140]]
    obstacles = obstacles1 + obstacles2 + obstacles3
    obstacles = np.array(obstacles)

    molecule_R = np.array([[10, 20], [100, 200], [200, 200], [50, 190], [10, 40]])
    final_R = np.array([[120, 120], [140, 120], [130, 110], [130, 130]])
    centX = 0
    centY = 0
    width = 20e-9
    height = 20e-9

    stm_img, V_arr, I_arr, z_arr, pos_arr, t_arr = automation_main(molecule_R, final_R, centX, centY, width, height, obstacles)

    # Plot observed data vs time
    plt.plot(t_arr, V_arr)
    plt.show()

    from matplotlib import animation

    # Animated plot of path of tip vs time
    fig, axes = plt.subplots(1, 3)
    axes[0].set_xlim(0 - width/2, 0 + width/2)
    path, = axes[0].plot([], [], lw = 1)
    I_line, = axes[1].plot([], [], lw = 1)
    z_line, = axes[2].plot([], [], lw = 1)

    def init():
        path.set_data([], [])
        I_line.set_data([], [])
        z_line.set_data([], [])

        return path, I_line, z_line,

    def animate(i):
        path.set_data(pos_arr[0:i, 0], pos_arr[0:i, 1])
        I_line.set_data(t_arr[0:i], I_arr[0:i])
        z_line.set_data(t_arr[0:i], z_arr[0:i])

        return path, I_line, z_line

    anim = animation.FuncAnimation(fig, animate, init_func = init,
        frames = np.arange(0, t_arr.shape[0], 1), interval = 20, blit = True)

    plt.tight_layout()
    plt.show()
