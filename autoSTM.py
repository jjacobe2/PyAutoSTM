''' PyAutoSTM

    Main module containing Nanonis TCP client class, for automating control of STM
    
    Juwan Jeremy Jacobe
    University of Notre Dame
    
    Last Updated: 30 Sept 2022
'''

# Internal imports
import tcp_commands as tcp

# External imports 
import numpy as np
import socket
import bitstring
import struct
from ctypes import *
import matplotlib.pyplot

class Nanonis:
    ''' Class data structure to handle TCP commands between Python and the Nanonis Software
    '''
    def __init__(self, ip_address = 'localhost', port = '6501'):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.time_delay = 1 # delay in seconds after sending command or receiving reply from Nanonis software
        
    def connect(self):
        self.sock.connect((ip_address, port))
        
    def close(self):
        self.sock.close()

if __name__ == "__main__":
    client = Nanonis()
    client.connect()
    
    num_channels, pixels, lines = tcp.get_scan_buffer(client)
    data = tcp.scan_frame_grab(client, 0, 1, lines, pixels)
    plt.imshow(data, cmap = 'hot')
    
    client.close()