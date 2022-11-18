''' stmmap.py --- PyAutoSTM

    Module containing STMMap object, which is a class containing information about a scan 
    and methods and functions for manipulating information about a scan

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 18 Nov 2022
'''
import numpy as np
import matplotlib.pyplot as plt
from autoutils.assignment import assignment
from autoutils.stmimaging import process_img, blob_detection, annihilate_blob

# Class handling data regarding an STM scan
class STMMap:
    ''' Class handling data regarding the primary STM scan and locations of molecules on the STM scan. 
    Important scan image paramaters:
        centX: physical x coordinate of center of scan frame
        centY: physical y coordinate of center of scan frame
        width: physical width of image in meters
        height: physical height of image in meters (typically: height = width, i.e. square image)
        angle: angle at which image is scanned at with respect to some x, y axes (kinda typically: angle = 0)
    '''

    def __init__(self, image_data, centX, centY, width, height, angle):

        # Store image and scan paramaters
        self.raw_image = image_data
        self.centX = centX # m
        self.centY = centY # m
        self.width = width # m
        self.height = height # m 
        self.angle = angle # degrees

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
        self.blob_detector = blob_detection

    # Method for processing/thresholding image
    def process_img(self, image, width, disp = False):
        self.processed_image = self.img_processor(image, width, disp)

    # Method for locating molecules, given a blob detection function
    def locate_molecules(self, image, width, disp = False):
        self.all_molecules_pixel = self.blob_detector(image, width, disp = disp)

        # Convert pixel coordinates to physical and store
        self.all_molecules = self.pixel2point(self.all_molecules_pixel.astype(int))

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

    # Image params
    pos_arr = np.array([[0, 0], [0.5, 0.5], [-0.1, 0.1]]) * 10e-9
    width = 20e-9
    bias_V = 0.01
    num_pixels = 256

    # Create image
    image = sc_sim.create_sim_topo_image(pos_arr, width, bias_V, num_pixels, 300)

    map = STMMap(image, 0, 0, width, width, 0)
    map.process_img(image, width, True) 

    new_img = annihilate_blob(map.processed_image.copy(), int(256/2), int(256/2))

    # Test out blob detection
    map.locate_molecules(map.processed_image, width, disp = True)
    print(map.all_molecules_pixel)
    print(map.all_molecules)