''' assignment.py

    Module containing functions for implementing an assignment algorithm using 
    scipy.optimize.linear_sum_assignment

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 26 Oct 2022
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Cost matrix built from position of initial configuration and positions of final configuration
class CostMatrix:
    ''' Class handling the creation a cost matrix given an initial array of N positions and a desired array of M final positions,
    with the cost criteria being the Euclidean distance between an initial point and a final point
    '''

    def __init__(self, init_R, final_R):
        ''' 
            Args:
                init_R (Nx2): numpy array of initial positions
                final_R (Mx2): numpy array of final positions
        '''
        
        self.init_R = init_R
        self.final_R = final_R
        self.N = init_R.shape[0]
        self.M = final_R.shape[0]

    def calc_distance(self, r_i, r_j):
        ''' Calculate distance between r_i and r_j, where both are 2-vectors of x, y
        '''

        dist = np.sqrt((r_i[0]-r_j[0])**2 + (r_i[1]-r_j[1])**2 )

        return dist

    def construct_matrix(self):
        ''' Construct matrix proper and populate entries with cost, i.e. the distance
        '''
        
        self.matrix = np.zeros((self.N, self.M))
        
        for i in np.arange(0, self.N, 1):
            for j in np.arange(0, self.M, 1):
                self.matrix[i, j] = self.calc_distance(self.init_R[i], self.final_R[j])

def assignment(init_R, final_R):
    ''' Given initial and final configuration, assign using linear_sum_assignment and return assigned positions

        Args:
            init_R (np.array): initial positions
            final_R (np.array): final goal positions

        Returns:
            assigned_init_R (np.array): entries of init_R that are assigned, ordered in such a way that it corresponds to assigned_final_R row by row
            assigned_final_R (np.array): entries of final_R that are assigned, ordered in such a way that it corresponds to assigned_init_R row by row
    '''

    # Create cost matrix
    cost_matrix = CostMatrix(init_R, final_R)
    cost_matrix.construct_matrix()

    # If row_ind[i] = j and col_ind[i] = k, then the jth dot (index j in init_R) is assigned to the kth goal location (index k in final_R)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.matrix)

    # Get assigned initial and final positions as matrices by indexing using row_ind and col_ind
    assigned_init_R = cost_matrix.init_R[row_ind]
    assigned_final_R = cost_matrix.final_R[col_ind]

    return assigned_init_R, assigned_final_R

if __name__ == "__main__":
    N = 20
    M = 10

    init_R = 1.2*np.random.rand(N, 2)
    #final_R = np.random.rand(M, 2)
    final_R = np.array([[0.45, 0.55], [0.55, 0.55], [0.45, 0.45], [0.55, 0.45], [0.5, 0.5]])

    assigned_init_R, assigned_final_R = assignment(init_R, final_R)

    fig = plt.figure()
    axes = fig.add_subplot()

    # Draw initial and final as scattered points
    axes.scatter(init_R[:, 0], init_R[:, 1], label = "Initial molecules")
    axes.scatter(final_R[:, 0], final_R[:, 1], label = "Desired configuration")

    # Draw lines between assigned initial and its corresponding final
    for i in np.arange(0, assigned_init_R.shape[0], 1):
        axes.plot([assigned_init_R[i, 0], assigned_final_R[i, 0]], [assigned_init_R[i, 1], assigned_final_R[i, 1]], color = "black")

    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    axes.set_aspect('equal')
    plt.tight_layout()
    plt.legend()
    plt.show()