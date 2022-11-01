''' stmmap.py --- PyAutoSTM

    Module containing STMMap object, which is a class containing information about a scan 
    and methods and functions for manipulating information about a scan

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 24 Oct 2022
'''
import numpy as np
from autoutils.assignment import assignment

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

    # Method for processing image, given a processing function
    def process_image(self, process_func):
        self.pro_image = process_func(self.raw_image)

    # Method for locating molecules, given a blob detection function
    def locate_molecules(self, blob_detector, image):
        self.all_molecules = blob_detector(image)

    # Method for defining a desired final configuration
    def define_final_configuration(self, pattern):
        self.final_config = pattern

    # Method for assigning molecules to the desired final configuration, given an assignment function
    def assign_molecules(self):
        self.assigned_init_config, self.assigned_final_config = self.assignment_func(self.all_molecules, self.final_config)

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
    pass