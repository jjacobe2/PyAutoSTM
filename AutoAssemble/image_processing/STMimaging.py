''' Module containing functions for processing images from STM scans. Planned functions
    
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
from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage.feature import blob_doh
import matplotlib.pyplot as plt

import pySPM
import pySPM.SXM as SXM

# Constants
CO_WIDTH_MIN = 1.00e-9   # The approximate width of CO molecule (m)
CO_WIDTH_MAX = 1.1e-9

''' Notes:
    Do pixel_threshold by comparing z values of bright and dark spots
'''

def read_SXM(img_name, plot_imgs=False, pixel_threshold=0.30, min_sigma=2.8284, max_sigma=2.85, threshold=0.015):
    ''' Function to read in and process an image and perform blob detection using the matrix of the Hessian determinant method 
        args:
            img_name (str): path + name of image to be read
        out:
            blobs (np.array): array where values are (y, x, sigma) where radius of blob is roughly sqrt(2)*sigma. Y and x
            seem to be pixel values perhaps. Note the +y direction is downwards and the +x direction is rightwards
            pixel_width (int): width in number of pixels
            pixel_height (int): height in number of pixels
    '''
    # Get image pixels
    sample = img_name
    
    # Contrast? Find better ways to do this
    average = np.average(2*sample)
    sample = np.where(sample < average, sample, sample-average)
    
    # Turn from colored --> grayscale
    # sample_g = rgb2gray(sample)

    # Binarize
    sample_b = sample > pixel_threshold
    
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
def find_molecules(img_name):
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
    
    image_curr = image_obj.get_channel(name='Current').pixels
    image_curr = pySPM.normalize(image_curr)
    
    print(image_curr)
    
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
    
    blobs = read_SXM(image, plot_imgs=True, min_sigma=min_sigma, max_sigma=max_sigma)
    points = blobs2points(blobs, centX, centY, width, height, pixel_width, pixel_height)
    
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
img_name = '../../sample2/Topo002.sxm'
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
1) Issues of CO molecules not having enough contrast from background
   --> Possible solution: "subtract" background image using a image of scanned background, where STM scans background
       with no CO molecule. Subtract image from image of interest to be left with CO molecules
'''

