''' planner.py

    Module containing functions to plan order of assembly such that paths
    aren't obstructed

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 9 Dec 2022 
'''

import numpy as np
import matplotlib.pyplot as plt

def arr2columns(mol_pos_arr: np.ndarray, start_x: float, end_x: float, dx: float = 1.5e-9) -> list:
    ''' Function to split an array of 2D position coordinates to a group of arrays representing those positions split up
    into vertical columns of width dx 

    Args:
        mol_pos_arr (np.ndarray): array of 2D positions
        start_x (float): x-position of most leftwards position
        end_x (float): x-position of most rightwards position
        dx (float): width of column (in meters)

    Return:
        columns (list): list of 2D arrays representing the columns, sorted so that first element is most left column
                        and last element is most right column 
    '''

    # Start with a column whose left bound is start_x subtracted with dx/2
    x_it = start_x - dx/2

    # Initiate empty list for the column arrays
    columns = []

    # Iterate until reach/or beyond end_x
    while x_it < end_x:

        # Initiate empty list for column indices
        column_indices = []

        # Check each x position of each molecule in mol_pos_arr. If within column, add its index to column indices
        # Note: THIS CAN BE MORE EFFICIENT BY REMOVING ELEMENT FROM MOL_POS_ARR SO THAT THE NEXT ITERATION DOESN'T HAVE TO 
        # ITERATE THROUGH THE WHOLE ARRAY. REMEMBER PYTHON FOR LOOPS ARE INEFFICIENT (ICKY)
        for i in np.arange(mol_pos_arr.shape[0]):
            if x_it < mol_pos_arr[i, 0] < x_it + dx:
                column_indices += [i]

        # Add column to columns
        columns += [mol_pos_arr[column_indices]]

        # Iterator
        x_it = x_it + dx

    return columns

def sortcolumns(columns: list) -> list:
    ''' Takes list of columns and sorts each column to be in order of increasing y-value

    Args:
        columns (list): list of arrays representing column of positions

    Returns:
        new_columns (list): list of arrays representing column, column sorted by y-position from lowest to highest 
    '''

    # Initiate empty list for sorted columns
    new_columns = []

    # Iterate through columns
    for column in columns:

        # Use np.argsort to get array of indices that would sort the array along axis = 0 (the axis going downwards
        # Index only the second column as interested in sorting by y-position
        sorted_indices = np.argsort(column, axis = 0)[:, 1]

        # Index column using sorted_indices to create sorted column
        sorted_column = column[sorted_indices]

        # Add sorted column to list
        new_columns += [sorted_column]

    return new_columns
    

def planner(mol_pos_arr: np.ndarray, dx: float = 1.5e-9) -> np.ndarray:
    ''' Function to sort x, y locations in mol_pos_arr so that it is built column by column from left to right and
    each column is built from bottom to top.

    First attempt at an algorithm to order the assembly of CO molecules such that there are no blockages preventing the program
    from getting a CO molecule to a desired final location
    '''

    # Find lowest x value and highest x value to create bounds
    start_x = np.amin(mol_pos_arr[:, 0])
    end_x = np.amax(mol_pos_arr[:, 0])

    # Split up into separate arrays representing vertical columns of width dx
    columns = arr2columns(mol_pos_arr, start_x, end_x, dx)

    # Sort each CO molecule location from highest to lowest y-value
    sorted_columns = sortcolumns(columns)

    # Rejoin columns in their sorted form and returns
    new_pos_arr = sorted_columns[0] # Take first column
    for i in np.arange(1, len(sorted_columns)):
        new_pos_arr = np.vstack((new_pos_arr, sorted_columns[i]))

    return new_pos_arr

if __name__ == "__main__":
    mol_pos_arr = np.array([[1, 0]])
    