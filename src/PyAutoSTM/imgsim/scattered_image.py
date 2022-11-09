''' scattered_image.py

    Module for generating high quality simulated images
    of an STM scan given a certain configuration of CO molecules on
    copper surface

    Juwan Jeremy Jacobe
    University of Notre Dame

    Translated from existing MATLAB code
    Scattering simulation functionalities credited to former graduate students in Dr. Kenjiro Gomes's group as well
    as Anthony Francisco and Nileema Sharma

    Last modified: 4 Nov 2022
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# CONSTANTS
E0 = 0.45 # Band edge for surface state electrons on Cu(111)
EC = 1.60217646e-19 # Electron charge (C)
HBAR = 6.58202e-16 # Reduced Planck's constant (eV s)

ME = 9.10938188e-31 # electron mass (kg)
MSTAR = 0.38 # effective mass for Cu(111) surface states electrons (ME)
M = MSTAR * ME # effective mass for Cu(111) surface state electrons (kg)

A = 2.54772 # Lattice constant of Cu(111) at 4K -- it is 2.55625 at 298K

# Confirmed to work exactly as kfit2Cu.m
def kfit2Cu(pos_arr, ntheta = 301, nsym = 1, a = 2.547, disp = False):
    ''' Adjust lattice coordinates vp to fit Cu lattice. 
    Lattice needs to be centered at (0, 0) which is a Cu site
    by default. Rotations are taken around (0, 0) with increments
    of 60def/ntheta

    NOTE: Need to find a way to go from lattice coordinates to real coordinates :(

    Args:
        pos_arr (np.ndarray, shape of (N, 2)): lattice coordinates of CO plan
        ntheta (int): the number of angles to search
        nsym (int): explore all angle space
        a (float): Cu lattice spacing
    '''

    # Determine image size based on pos_arr
    L = np.sqrt(np.max(np.power(pos_arr[:, 0], 2) + np.power(pos_arr[:, 1], 2)))
    N_x = int(np.ceil(L/a) + 2)
    N_y = int(np.ceil(N_x/np.sqrt(3)))

    # Create copper sites. Build a (2*N_x + 1)(2*N_y + 1)*2 by 2 array of Cu sites
    # currently doing dynamically using lists instead of preallocating using np.ndarray
    # cu_sites = np.zeros((2*N_x + 1)*(2*N_y + 1)*2, 2)
    cu = []

    for n_j in np.arange(-N_y, N_y + 1, 1):
        for n_i in np.arange(-N_x, N_x + 1, 1):
            v_1 = [[a*n_i, a*n_j*np.sqrt(3)]]
            v_2 = [[v_1[0][0] + a * np.cos(np.pi/3), v_1[0][1] + a * np.sin(np.pi/3)]]

            # v_1 and v_2 as nested lists --> i.e. as individual rows
            cu = cu + v_1 + v_2

    # Convert to numpy array
    cu = np.array(cu)

    # Do fitting for various angles to find minimum
    theta = np.linspace(0, np.pi / (3*nsym), ntheta)
    RMSE = np.zeros((ntheta, 1))
    min_err = 10**10

    for n_t in np.arange(0, ntheta, 1):

        # For this value of theta...
        t = theta[n_t]

        # ... construct a rotation matrix ...
        M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

        # ... and apply this rotation to the CO plan
        pos_arr_rotated = np.transpose(M @ np.transpose(pos_arr))

        # Calculate distance from co_plan to cu sites
        X1, X2 = np.meshgrid(pos_arr_rotated[:, 0], cu[:, 0])
        Y1, Y2 = np.meshgrid(pos_arr_rotated[:, 1], cu[:, 1])
        D = np.sqrt(np.power(X1 - X2, 2) + np.power(Y1 - Y2, 2))

        # Calculate closest co site, translate matlab's min to Python, not exact
        # so kinda have to finagle it like this. This whole block is equivalent to [allerror, cusites] = min(D)
        allerror = np.zeros(D.shape[1])
        cusites = np.zeros(D.shape[1]) 
        for n_col in np.arange(0, D.shape[1], 1):
            allerror[n_col] = np.min(D[:, n_col])
            cusites[n_col] = int(np.argmin(D[:, n_col]))

        # Calculate RMSE and compare with current minimum error
        RMSE[n_t] = np.linalg.norm(allerror)
        if RMSE[n_t] < min_err:
            min_err = RMSE[n_t]
            pos_arr_Cu = cu[cusites.astype(int), :]
            theta0 = t
            pos_arr_rot_final = pos_arr_rotated

    # Normalizing by number of atoms
    num_atoms = np.sqrt(pos_arr.shape[0])
    RMSE = RMSE / num_atoms
    min_err = min_err / num_atoms

    ## Optional display stuff
    if disp:

        # plot RMSE vs various theta
        fig = plt.figure()
        axes = plt.axes()
        axes.set_xlabel('theta (degrees)')
        axes.set_xlim(0, 60/nsym)
        axes.set_ylabel('MSE')
        axes.set_title('Angular Error Dependence')
        axes.plot(theta * 180/np.pi, RMSE)
        plt.show()

        # plot result
        fig = plt.figure()
        axes = plt.axes()
        axes.set_xlim(-20, 20)
        axes.set_ylim(-20, 20)
        axes.set_aspect('equal')
        axes.scatter(cu[:, 0], cu[:, 1], marker = ',', color = 'black', s = 2)
        axes.scatter(pos_arr_rot_final[:, 0], pos_arr_rot_final[:, 1], marker = 'x', color = 'red', s = 15)
        axes.scatter(pos_arr_Cu[:, 0], pos_arr_Cu[:, 1], marker = 'o', color = 'blue', s = 5)
        plt.show()

    return pos_arr_Cu, theta0, RMSE

def kv2k(V, dispersion = [], casenumber = 0):
    ''' Returns wavenumber for quasiparticle tunneling into Cu(111)
    where V is the voltage of the sample with respect ot the tip

    Units: 1/Angstrom.

    Matches data taken on a wall of CO molecules from 9 - 6 - 2017
    '''

    if casenumber == 1:
        # Old code
        if len(dispersion) == 0:
            alpha = 0

        elif len(dispersion) > 1:
            if len(dispersion) == 3:
                alpha = dispersion[2]

            # Recalculate effective mass
            MSTAR = dispersion[1]
            M = MSTAR * ME
            E0 = dispersion[0]

        energy = V + E0

        # Evaluate inverse dispersion relation
        if alpha == 0:
            k = np.sqrt(energy / EC * (2 * M)) / HBAR * 1e-10
        else:
            b = HBAR ** 2 / (2 * M) * 10**20 * EC
            k = np.sqrt((-b + np.sqrt(b**2 + 4*alpha*energy))/(2*alpha))

    else:
        # Coefficients from fitting data
        # Dispersion relation this is using is energy = a1 + b1*(k**2) + c1*(k**4)
        a1 = -0.44030822387839125
        #b1 = 0.09385747589952433
        b1 = 0.11
        c1 = -0.0011500761131047537
        
        if (0.1032 -(b1/(c1*2)) + ((np.sqrt(b1**2 + 4*V*c1 - 4*a1*c1))/(c1*2))) < 0:
            print('WARNING: INCORRECT RESULTS ARE BEING SHOWN BECAUSE ENERGY IS LESS THAN -0.45V')
        
        k = np.real(np.sqrt(-(b1/(c1*2)) + ((np.sqrt(b1**2 + 4*V*c1 - 4*a1*c1))/(c1*2)))) / 10

    return k

def kmap(scatter_pos_arr, map_size, bias_V = 0, num_pix = 200, delta = None):
    ''' Calculate the LDOS map for given set of scatterers

    Args:  
        scatter_pos_arr (np.ndarray): array with the position of the scatterers, in angstroms?
        map_size (float): size of the map in angstroms(?)
        bias_V (float): bias voltage in V (0 by default)
        num_pix (int): number of pixels of the map (200 by default)
        delta (float/complex): phase change in the scattering

    Return:
        LDOS (2D np.ndarray): image/map of CO scatterers on Cu
    '''

    # Set default values for params
    if delta == None:
        f = -1/2 # for case where delta = i * infinity
    else:
        f = (np.exp(2 * 1j * delta) - 1)/2

    if bias_V == 0:
        k = 0.21185
    else:
        k = kv2k(bias_V)

    x = np.linspace(-map_size/2, map_size/2, num_pix)
    X = np.array(np.meshgrid(x))
    Y = np.transpose(X)

    n_sc = scatter_pos_arr.shape[0] # number of scatterers 

    # Build up lookup table for bessel functions of third kind (Hankel functions of first kind)
    # Speeds up the calculations
    res = 0.01
    max_r = np.sqrt(2)*map_size
    Htable = scipy.special.hankel1(0, np.arange(0, max_r, res)* k)
    Htable[0] = 0

    # Compute distances between pairs of atoms
    prd = scatter_pos_arr @ np.transpose(scatter_pos_arr)
    sqr = np.array([np.diagonal(prd)])
    RS = np.sqrt(np.transpose(sqr) @ np.ones((1, n_sc)) + np.ones((n_sc, 1)) @ sqr - 2*prd)

    # Calculate Re[f*h*(1-f*H)**-1 * h]
    A = f * Htable[np.rint(RS/res).astype(int)]
    A[np.isnan(A)] = 0
    B = np.linalg.inv( np.eye(n_sc) - A )

    # This for loop is faster than some 3D meshgrid
    r = np.zeros((num_pix, num_pix, n_sc))

    for i in np.arange(0, n_sc, 1):
        r[:, :, i] = np.sqrt( np.power(X - scatter_pos_arr[i, 0]*np.ones(X.shape), 2) + np.power(Y - scatter_pos_arr[i, 1]*np.ones(Y.shape), 2))
    
    # How the following works
    # H0tensor will describe the propagation from each point in the scan (from the
    # stm tip) to an atom. Each page will be for a different atom
    htensor = Htable[np.rint(r/res).astype(int)]

    # TO vectorize, reshape Htensor, to make a matrix where
    # each column corresponds to an atom and each scan point is a row
    htensor = np.reshape(htensor, (int(num_pix**2), n_sc))

    # Compute Green's function by computing the propagation from the tip to an
    # atom, between the atoms, and back to the tip. The sum adds the contributions
    # from starting with each atom
    C = np.sum(np.multiply((htensor @ B), htensor), axis = 1)
    C = np.reshape(C, (num_pix, num_pix))

    LDOS = np.real(f*C) + np.ones(C.shape)

    return LDOS

def create_sim_image(mol_pos_arr, width, bias_V, num_pixels):
    ''' Abstracted function to create a simulated STM image of CO molecules on Cu(111) using scattering simulation
    
    Args:
        mol_pos_arr (np.ndarray): a N x 2 array of CO molecules coordinates (m)
        width (float): width/height of the image (m)
        bias_V (float): desired bias voltage to simulate stm image under (V)
        num_pixels (int): pixel width and pixel height of image

    Return:
        img (np.ndarray): a num_pixels x num_pixels array representing the STM image
    '''

    # Convert pos_arr and width from being in units of meters to angstroms
    mol_pos_arr = mol_pos_arr * 1e10
    width = width * 1e10

    print(mol_pos_arr)
    print(width)
    # Get image
    img = kmap(mol_pos_arr, width, bias_V, num_pixels)

    return img

if __name__ == "__main__":

    # Copying from Tony's "Lauras_Stuff.m example"
    # Create molecular graphene
    pos_arr = []
    #for i in np.arange(0, 6, 1):
    #    pos_arr = pos_arr + [[20*np.cos(i* 2*np.pi/6 - 2*np.pi/6), 20*np.sin(i * 2*np.pi/6 - 2*np.pi/6)]]
    #for i in np.arange(0, 6, 1):
    #    pos_arr = pos_arr + [[10*np.cos(i* 2*np.pi/6 - 1*np.pi/6), 10*np.sin(i * 2*np.pi/6 - 1*np.pi/6)]]
    pos_arr = pos_arr + [[0, 0]]
    pos_arr = pos_arr + [[0, 10]]
    pos_arr = np.array(pos_arr)
    pos_arr_Cu, t0, rmse = kfit2Cu(pos_arr, disp = False)

    mapsize = 100
    delta=0.2*(-1+1j) # value set by Tony
    LDOS = kmap(pos_arr, mapsize, 0.5, 256, delta) ** 2

    plt.imshow(LDOS, cmap = 'gray', vmin = 0, vmax = 2)
    plt.clim(0.6, 1.6)
    plt.show()

    from skimage.filters import threshold_minimum

    # bin_LDOS = LDOS > threshold_minimum(LDOS)
    # plt.imshow(filt_LDOS, cmap = 'gray')
    plt.show()