''' stmmap.py --- PyAutoSTM

    Module containing STMMap object, which is a class containing information about a scan 
    and methods and functions for manipulating information about a scan

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 8 Nov 2022
'''
import numpy as np
from autoutils.assignment import assignment
from autoutils.stmimaging import process_img

# Class handling data regarding an STM scan
class STMMap:
    ''' Class handling data regarding the primary STM scan and locations of molecules on the STM scan
    '''
    def __init__(self, image_data, centX, centY, width, height, angle):
        self.raw_image = image_data
        self.centX = centX
        self.centY = centY
        self.width = width
        self.height = height
        self.angle = angle

        # Calculate width and height of images in pixels by looking at dimensions of image array
        self.pixel_height, self.pixel_width = self.raw_image.shape

        # Find dx and dy with respect to pixel dimensionality of image
        self.dx = self.width/self.pixel_width
        self.dy = self.height/self.pixel_height

        # Find the physical coordinates of top left corner of image to use
        # as reference
        self.top_left_x = self.centX - self.width/2
        self.top_left_y = self.centY + self.height/2

        # Save assignment algorithm as object belonging to class
        self.assignment_func = assignment
        self.img_processor = process_img

    # Method for processing/thresholding image
    def process_img(self, image, width):
        self.processed_image = self.img_processor(image, width)

    # Method for locating molecules, given a blob detection function
    def locate_molecules(self, blob_detector, image):
        self.all_molecules = blob_detector(image)

    # Method for defining a desired final configuration
    def define_final_configuration(self, pattern):
        self.final_config = pattern

    # Method for assigning molecules to the desired final configuration, given an assignment function
    def assign_molecules(self):
        self.assigned_init_config, self.assigned_final_config, self.assigned_indices = self.assignment_func(self.all_molecules, self.final_config)
        self.unmoved_all_molecules = self.all_molecules
        self.unmoved_init_molecules = self.assigned_init_config
        self.unfilled_final_molecules = self.assigned_final_config

    # Method for confirming a move and removing a molecule from variable for storing specifically the assigned molecules that haven't been moved
    # and for also storing final molecule positions that haven't been "filled" yet
    def confirm_successful_move(self, index):
        self.unmoved_all_molecules = np.delete(self.unmoved_all_molecules, self.assigned_indices[index], axis = 0)
        self.unmoved_init_molecules = np.delete(self.unmoved_init_molecules, index, axis = 0)
        self.unfilled_final_molecules = np.delete(self.unfilled_final_molecules, index, axis = 0)

    # insert function here doing pixel --> physical transformation & vice versa
    def pixel2point(self, pixel_arr):
        ''' Transform pixels coordinates to physical point coordinates in an
        image given parameters of STMMap image

        Args:
            pixel_arr (Nx2): array of pixel coordinates

        Return:
            points_arr (Nx2): array of physical coordinates
        '''
        
        # Grab pixel_x and pixel_y as separate Nx1 array from pixel_arr
        pixel_x = pixel_arr[:, 1]
        pixel_y = pixel_arr[:, 0]

        # Change pixel number ---> x and y
        points_x = self.top_left_x * np.ones(pixel_x.shape) + self.dx * pixel_x
        points_y = self.top_left_y * np.ones(pixel_y.shape) - self.dy * pixel_y # Second term is negative as pixel number is counted down while y is positive up

        points_arr = np.stack((points_x, points_y), axis = -1)

        return points_arr

    def point2pixel(self, points_arr):
        ''' Transform physical point coordinates to pixel coordinates in an
        image given parameters of STMMap image

        Args:
            points_arr (Nx2): array of physical coordinates

        Return:
            pixel_arr (Nx2): array of pixel coordinates
        ''' 

        # Grab points_x and points_y as separate Nx1 array from points_arr
        points_x = points_arr[:, 0]
        points_y = points_arr[:, 1]

        # Change x, y --> pixel number
        pixel_x = np.array((points_x - self.top_left_x * np.ones(points_x.shape)) / self.dx).astype(int)
        pixel_y = np.array(((self.top_left_y * np.ones(points_x.shape) - points_y) / self.dy )).astype(int)

        pixel_arr = np.stack((pixel_y, pixel_x), axis = -1)

        return pixel_arr

if __name__ == "__main__":
    import imgsim.scattered_image as sc_sim
    #pos_arr = np.array([[0, 0], [0.5, 0.5], [-0.25, 0.75], [-0.05, 0.1],]) * 10e-9
    pos_arr = np.random.uniform(low = -1, high = 1, size =(20, 2))* 10e-9
    width = 20e-9
    bias_V = 0.1
    num_pixels = 256

    image = sc_sim.create_sim_image(pos_arr, width, bias_V, num_pixels)

    map = STMMap(image, 0, 0, width, width, 0)
    map.process_img(image, width) 
