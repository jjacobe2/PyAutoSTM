''' autocmd.py

    Module for custom composite TCP commands built from combining the building block commands
    from tcp.py. Primarily functions that pass a Nanonis object as an argument

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last Updated: 30 Nov 2022
'''

import time
import numpy as np
from tcp import Nanonis
from stmmap import STMMap

# Follow path
def follow_path(stm: Nanonis, stm_map: STMMap, path_arr: list = None, V_arr: list = None, I_arr: list = None, 
    z_arr: list = None, pos_arr: list = None, t_arr: list = None, t0: float = None, sampling: bool = False):
    ''' Function for handling tcp commands to make stm tip follow a certain path

    Args:
        stm (Nanonis): the TCP client connected to the Nanonis SPM Controller software
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
        stm.folme_xyposset(point[0], point[1], wait_end_of_move = 1)
        if sampling:
            V_arr = V_arr + [stm.bias_get()]
            I_arr = I_arr + [stm.current_get()]
            z_arr = z_arr + [stm.zctrl_zposget()]
            t_arr = t_arr + [time.time() - t0]
            x, y = stm.folme_xyposget(wait_for_new = 0)
            pos_arr = pos_arr + [[x, y]]

    return V_arr, I_arr, z_arr, pos_arr, t_arr

def scan_mode(stm: Nanonis, bias_V: float, setpoint: float, tip_speed: float):
    ''' Set Nanonis to scan mode by changing bias, changing current set point and changing tip move speed

    Args:
        stm (Nanonis): the TCP client connected to the Nanonis STM Controller software
        bias_V (float): bias voltage (V)
        setpoint (float): setpoint current (A)
        tip_speed (float): speed of tip (m/s)
    '''

    stm.bias_set(bias_V)
    stm.zctrl_setpntset(setpoint)
    stm.folme_speedset(tip_speed, custom_speed_mod = 0) # Setting second argument is basically saying use scan speed, so first arg doesn't actually matter lol

# Identical to scan_mode, but just two different funcs for sake of separation --- redo this another way l8r
def manip_mode(stm: Nanonis, bias_V: float, setpoint: float, tip_speed: float):
    ''' Set Nanonis to manipulation mode by changing bias, changing current set point and changing tip move speed

    Args:
        stm (Nanonis): the TCP client connected to the Nanonis STM Controller software
        bias_V (float): bias voltage (V)
        setpoint (float): setpoint current (A)
        tip_speed (float): speed of tip (m/s)
    '''
    
    stm.bias_set(bias_V)
    stm.zctrl_setpntset(setpoint)
    stm.folme_speedset(tip_speed, custom_speed_mod = 1) # Use given tip_speed

# Scan with specific settings and give data from channel