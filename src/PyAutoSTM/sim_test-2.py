''' sim_test-2.py

    Script for running for automated assembly procedure -- similar to sim_test-1.py -- but
    with image processing and blob detection integrated, using scattering simulations in order
    to generate synthetic STM images of CO molecules on Cu(111)

    Test meant to run with Nanonis STM Simulator v5

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 8 Nov 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import time

import autoutils.astar as astar
from imgsim.scattered_image import create_sim_image
from stmmap import STMMap
from tcp import Nanonis

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

if __name__ == "__main__":
    img_width = 20e-9
    bias_V = 0.5
    num_pixels = 256
    molecules_pos_arr = np.array([[]])