''' Module containing functions for processing images from STM scans. Planned funcastions
    
    readSXM:
        args:
            file_name (str): path + name of image to be read
        out:
            blobs (np.array): array where values are (y, x, sigma) where radius of blob is roughly sqrt(2)*sigma. Y and x
            seem to be pixel values perhaps. Note the +y direction is downwards and the +x direction is rightwards
            
    blobs2points
        args:
            blobs (np.array): the output from readSXM
            centX (float): x coordinate of center of image (m)
            centY (float): y coordinate of center of image (m)
            width (float): width of image (m)
            height (float): height of image (m)
            pixel_width (int): numbers of pixels across width of image
            pixel_height (int): numbers of pixels across height of image
            
        out:
            points (nd.array): array where values are (x, y), where x and y are the coordinates corresponding to the center
            of each found blob
            
    This current iteration requires pySPM in order to read SXM files
'''

import numpy as np
import skimage
import pySPM
import pySPM.SXM as SXM
import matplotlib.pyplot as plt

from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.feature import blob_doh

import astar

# Constants
CO_WIDTH_MIN = 0.90e-9   # The approximate width of CO molecule (m)
CO_WIDTH_MAX = 1.1e-9

''' Notes:
    Do pixel_threshold by comparing z values of bright and dark spots
'''

def process_img(image, global_thres=True):
    ''' Function to process and binarize image
    '''

    # Get image pixels
    sample = image
    
    # Contrasting? 
    average = np.average(2*sample)
    
    sample = np.where(sample > average, sample, sample-average)
    
    if global_thres:
        # Use Otsu's method for pixel thresholding?
        pixel_threshold = skimage.filters.threshold_otsu(sample)
    else:
        # Use local thresholding
        pixel_threshold = skimage.filters.threshold_li(sample)
        
    # Binarize
    sample_b = sample > pixel_threshold
    
    return sample_b
    
def read_SXM(image, plot_imgs=False, min_sigma=2.8284, max_sigma=2.85, only_free_molecules=True):
    ''' Function to read in and process an image and perform blob detection using the matrix of the Hessian determinant method 
        args:
            img_name (str): path + name of image to be read
        out:
            blobs (np.array): array where values are (y, x, sigma) where radius of blob is roughly sqrt(2)*sigma. Y and x
            seem to be pixel values perhaps. Note the +y direction is downwards and the +x direction is rightwards
            pixel_width (int): width in number of pixels
            pixel_height (int): height in number of pixels
    '''
    
    if only_free_molecules:
        threshold = 0.045
    else:
        threshold = 0.025
    
    sample = image 
    
    # Binarize
    sample_b = process_img(image)

    # To detect abnormal blobs, do blob detection, with min_sigma_normal = 2.5*min_sigma_normal and max_sigma = 3*max_sigma_normal
    
    # Now determine blobs using matrix of Hessian determinant method
    # Note that blob is returned as an n by 3 numpy array representing 3 values (y, x, sigma). (y, x) are coordinates of the
    # blob and sigma is the standard deviation of the Gaussian kernel of the Hessian Matrix whose determinant detected the blob.
    #   -> The radius of each blob is approximately sqrt(2)*sigma
    blobs = blob_doh(sample_b, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    
    if plot_imgs:
    
        # Create subplots
        fig, ax = plt.subplots(1,3,figsize=(10,5))
        ax[0].imshow(sample)
        ax[1].imshow(sample_b,cmap='gray')
        ax[2].imshow(sample_b,cmap='gray')
        ax[0].set_title('Colored Image',fontsize=15)
        ax[1].set_title('Binarized Image',fontsize=15)
        ax[2].set_title('Blob Detection')
        
        # Create circles signifiying blob locations
        for blob in blobs:
            y, x, area = blob
            ax[2].add_patch(plt.Circle((x, y), area*np.sqrt(2), color='r', 
                                fill=False))
                                
        plt.tight_layout()
        plt.show()
    
    return blobs #, pixel_width, pixel_height

def path_finder(image, start, end, CO_pixels, coordinate='pixels', visualize=True, verbose=False):

    # Binarize
    sample_b = process_img(image)
    
    # Get binarized image and turn from False, True --> 1, 0
    sample_graph = sample_b
    sample_graph = np.where(sample_graph == True, 0, 1)
    
    sample_graph_pro = sample_graph.copy()

    for i in np.arange(0, sample_graph.shape[0], 1):
        for j in np.arange(0, sample_graph.shape[1], 1):
        
            # If pixel on "graph" turned on, make neighbours turned on by setting to 1 as well. The purpose of
            # this is to make the path guaranteed not to get near any obstacles
            if sample_graph[i][j] == 1:
            
                for n in np.arange(-int(CO_pixels/1.5), int(CO_pixels/1.5), 1):
                    for m in np.arange(-int(CO_pixels/1.5), int(CO_pixels/1.5), 1):
                        if 0 < i + n < sample_graph.shape[0] and 0 < j + m < sample_graph.shape[1]:
                            sample_graph_pro[i+n][j+m] = 1  
                           
    
    plt.imshow(sample_graph_pro)
    plt.show()
    
    # Use A* path finding algorithm
    path = astar.search(sample_graph_pro, 1, start, end)
    if visualize:
    
        # Make image of the path 
        path_visualized = np.ones(sample_b.shape)
        
        for node in path:
            path_visualized[node] = 0
        
        plt.figure()
        
        plt.imshow(sample_b)
        plt.imshow(path_visualized, cmap='gray', alpha=0.5)
        plt.title(f'Path between {start} and {end}')
        plt.xlabel(f'Pixels')
        plt.ylabel(f'Pixels')
        plt.show()
    
    return path

def get_edges(image, plot=False):
    ''' Function to get edges from image. Let's use this for our pathfinding algorithm maybe?
     
    Args
        img (np.array): array representing binarized image
    
    Out
        edges (np.array): array representing edges of image
    ''' 
    
    edge_roberts = skimage.filters.roberts(image)
    
    if plot:
        plt.imshow(edge_roberts)
        plt.show()

    return edge_roberts
    
def blobs2points(blobs, centX, centY, width, height, pixel_width, pixel_height):
    ''' Function to take pixel numbers/locations and turn into physical coordinates in space in meters, given
    an image width and height + coordinates for the center of the image
        args:
            blobs (np.array): the output from readSXM
            centX (float): x coordinate of center of image (m)
            centY (float): y coordinate of center of image (m)
            width (float): width of image (m)
            height (float): height of image (m)
            pixel_width (int): numbers of pixels across width of image
            pixel_height (int): numbers of pixels across height of image
            
        out:
            points (nd.array): array where values are (x, y), where x and y are the coordinates corresponding to the center
            of each found blob-
    '''
    
    # Grab x and y pixel numbers from blobs array
    points_x = blobs[:, 1]
    points_y = blobs[:, 0]
    
    # Find the physical coordinate of top left corner of image to use as reference
    top_left_x = centX - width/2
    top_left_y = centY + height/2
    
    # Find dx and dy with respect to pixel number
    dx = width/pixel_width
    dy = height/pixel_height
    
    # Change pixel number --> x and y
    points_x = top_left_x*np.ones(points_x.shape) + dx*points_x
    points_y = top_left_y*np.ones(points_y.shape) - dy*points_y # second term is negative as pixel number is counted down while y is positive up
    
    points = np.stack((points_x, points_y), axis=-1)
    
    return points 

# Smart Read, calculates min_sigma and max_sigma based on image size
def find_molecules(img_name, only_free_molecules=True):
    ''' Function to call readSXM with specific min and max sigma according to width and height of the
    image to generalize the blob detection algorithm and then call points to get physical coordinates of found
    CO molecules
    
    Args:
        img_name (str): path + name of image being read
        
    Out:
        points (np.array): array of physical coordinates of CO molecules
    '''
    
    # Returns an SXM object which contains images from different channels + metadata embedded in the file
    image_obj = SXM(img_name)
    
    # Grab image with z channel, get pixels and then normalize to be able to threshold properly for algorithm
    image_z = image_obj.get_channel(name='Z').pixels
    image_z = pySPM.normalize(image_z)
    
    # Get channel for current as well and normalize
    image_curr = image_obj.get_channel(name='Current').pixels
    image_curr = pySPM.normalize(image_curr)
    
    # Subtract normalized current channel from normalized z channel to get a "contrasted" image
    image = pySPM.normalize(image_z - image_curr)
    
    # Pull info, header returns a dictionary which contains metadata about the file
    image_info = image_obj.header
    
    # Get value using key, unwrap double wrapped list, and pull values
    scan_area_info = image_info['Scan>Scanfield'][0][0].split(';') # image_info['Scan>Scanfield'][0][0] is string of numbers separated by semicolons, split to list using split method
    centX = float(scan_area_info[1])
    centY = float(scan_area_info[0])
    width = float(scan_area_info[3])
    height = float(scan_area_info[2])
    
    pixel_width = int(image_info['SCAN_PIXELS'][0][1])
    pixel_height = int(image_info['SCAN_PIXELS'][0][0])
    
    # Calculate min_sigma and max_sigma based on images sizes
    CO_pixel_width_min = int(CO_WIDTH_MIN/width * pixel_width)
    CO_pixel_width_max = int(CO_WIDTH_MAX/width * pixel_width)
    min_sigma = CO_pixel_width_min/(2*np.sqrt(2))
    max_sigma = CO_pixel_width_max/(2*np.sqrt(2))
    
    # Get blobs and convert to points
    blobs = read_SXM(image, plot_imgs=True, min_sigma=min_sigma, max_sigma=max_sigma, only_free_molecules=only_free_molecules)
    points = blobs2points(blobs, centX, centY, width, height, pixel_width, pixel_height)
    
    # Test pathfinder, erase later my guy!!!!!
    path = path_finder(image, (0, 100), (250, 50), CO_pixel_width_max)
    
    return points

def Hungarian_assign(points, targets):
    ''' Use the Hungarian method/ Munkres's algorithm to assign the various points in the scan frame to the points
    in the target configuration
    
    Args:
        points (np.array): array of x and y locations of molecules
        targets (np.array): array of x and y target locations of molecules desired configuration
        
    Out:
        molecules (np.array): n by 4 np array, where first two are origin locations and second two are target locations
    '''
    
    # Construct 2d matrix to perform algorithm on, where row spans the final configuration spots, column spans the
    
    pass


# Try out on images
img_name = '../../sample2/Topo014.sxm'
points = find_molecules(img_name)

ax = plt.axes()
ax.scatter(10**9*points[:, 0], 10**9*points[:, 1], s = 1)
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
ax.set_title("Physical Locations of Detected Blobs")
plt.show()

'''
Comments: 

Action Items
1) Start to do Hungarian assignment
2) Do a more comprehensive, parallel path assignment + procedure of autoassembly
   --> Copy/Paste Molecules to fake atom manipulation
   --> Finally implement all this in LabView
3) Clean up Procedure
4) Check if molecule manipulated successfully
'''

