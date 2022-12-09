''' planner.py

    Module containing functions to plan order of assembly such that paths
    aren't obstructred

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 9 Dec 2022 
'''

import numpy as np
import matplotlib.pyplot as plt

def planner(mol_pos_arr: np.ndarray, dx: float = 1.5e-9):
    ''' 
    '''

    # Find lowest x value and highest x value to create bounds
    
    # Decide width/tolerance (for now done manually)

    # Split up into separate arrays representing vertical columns of width dx

    # Sort each CO molecule location from highest to lowest y-value

    # Rejoin columns in their sorted form and returns
