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
from imgsim.scattered_image import create_sim_topo_image
from stmmap import STMMap
from tcp import Nanonis

# Settings
BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

# Function main 
def automation_main(stm_image, desired_pos_arr, centX, centY, width, plot_process = False):
    ''' 
    '''

    # Create STMMap object
    stm_map = STMMap(image, centX, centY, width, width, 0)

    # Blob detection
    stm_map.process_img(image, width, disp = True) # Process image
    stm_map.locate_molecules(stm_map.processed_image, width, disp = True) # Detect blobs

    # Hungarian Assignment
    stm_map.define_final_configuration(desired_pos_arr)
    stm_map.assign_molecules()

    # Primary loop -- start with initiating stm

    # Scanning mode

    # Move tip

    # Manipulation mode
    # Slow
    # Slowly move tip to final
    # Scanning mode
    # Scan/i.e. regenerate new image
    # Do image processing to see if moved successfully
    
if __name__ == "__main__":

    # Create image
    img_width = 20e-9
    bias_V = 0.5
    num_pixels = 256
    integration_points = 150
    molecules_pos_arr = np.array([[7, 3], [5, 5], [9, 7], [3.5, 5], [4, -2], [-4, -4], [5, -5], [-7, -7], [-5, 6]]) * 1e-9

    image = create_sim_topo_image(molecules_pos_arr, img_width, bias_V, num_pixels, integration_points)
    desired_pos_arr = np.array([[2, -2], [2, 2], [-2, 2], [-2, -2]])

    centX = 0
    centY = 0
    width = img_width

    automation_main(image, desired_pos_arr, centX, centY, width, plot_process = True)