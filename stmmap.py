''' stmmap.py --- PyAutoSTM

    Module containing STMMap object, which is a class containing information about a scan 
    + methods and functions for manipulating information about a scan

    Juwan Jeremy Jacobe
    University of Notre Dame

    Last modified: 24 Oct 2022
'''
import numpy as np

# insert function here doing pixel --> physical transformation & vice versa

# Class handling data regarding an STM scan
class STMMap:
    def __init__(self, image_data, centX, centY, width, height, angle):
        self.raw_image = image_data
        self.centX = centX
        self.centY = centY
        self.width = width
        self.height = height
        self.angle = angle

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
    def assign_molecules(self, assignment_alg):
        self.init_config = assignment_alg(self.all_molecules, self.final_config)

if __name__ == "__main__":
    pass