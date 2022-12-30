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
from stmmap import STMMap
from tcp import Nanonis

# Settings
BIAS_SCAN = 10e-3 # 10 mV
SETPOINT_SCAN = 500e-12 # 500pA
BIAS_MANIP = 10e-3 # 10 mV
SETPOINT_MANIP = 80e-9 # 80nA
SPEED_MANIP = 1e-9 # 1nm/s

# For generating desired pos array
def generate_square_arr(centX: float, centY: float, spacing: float) -> np.ndarray:
    ''' Generate position array for 5 x 5 Cs square, centered on (centX, centY)

    Args:
        centX (float): x-position of center of square
        centY (float): y-position of center of square
        spacing (float): spacing between Cs atoms

    Out:
        square_arr (np.ndarray): position array for Cs atoms in the square
    '''

    # Initiate
    square_arr = []

    # Generate top-line and bottom line
    for i in np.arange(-2, 3, 1):
        square_arr = square_arr + [[centX + i * spacing, centY + 2 * spacing]] + [[centX + i * spacing, centY - 2 * spacing]]

    # Generate side lines
    for i in np.arange(-1, 2, 1):
        square_arr = square_arr + [[centX + 2 * spacing, centY + i * spacing]] + [[centX - 2 * spacing, centY + i * spacing]]

    square_arr = np.array(square_arr)

    return square_arr

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

if __name__ == "__main__":

    # Test generate square arr
    centX = 0
    centY = 0
    spacing = 1.5e-9
    square_arr = generate_square_arr(centX, centY, spacing)

    plt.scatter(square_arr[:, 0], square_arr[:, 1])
    plt.show()
