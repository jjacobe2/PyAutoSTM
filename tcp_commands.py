import numpy as np
import socket
import bitstring
import struct
from ctypes import *

def append_command(command, len_max):
    new_command = command
    
    while (len(new_command) < len_max):
        new_command = new_command + b'\x00'
        
    return new_command

## Put all the type conversions to a different file --- maybe called tcp_utils for good practice? Just store the Nanonis class to here to make and carry out commands
## Maybe store the actual commands themselves in different files as well for program organization
# Note, every integer/float/double byte representation is in big-endian representation

# float to hex
def float2hex(f_val):
    h = struct.pack('>f', f_val)
    
    return h

# hex to float
def hex2float(hex_string):
    f_val = struct.unpack('>f', hex_string)
    
    return f_val

# double to hex
def double2hex(d_val):
    h = struct.pack('>d', d_val)
    
    return h
    
# hex to double
def hex2double(hex_string)
    d_val = struct.unpack('>d', hex_string)
    
    return d_val

## Do we need this for what we need so far?
# hex to 1D array 
# 1D array to hex
# hex to 2D array
# 2D array to hex
 
def int2byte(val, size = None):
    ''' Function to convert ints to byte strings
    '''
    if size == 16:
        # 16 bit, strip the 0x part
        msg_hex = f"{val:0{4}x}"
    
    else:
        # 32 bit (default), strip the 0x part
        msg_hex = f"{val:0{8}x}"
    
    # Define h as an array with size = number of bytes of the full message
    size = int(len(msg_hex)/2)
    h = bytearray(size)
    
    # Fill h array with the hex bytes
    for k in range(size):
        t = bitstring.BitArray(hex = msg_hex[2*k:2*k+2])
        h[k] = t.uint # Converts the bit array to uint, unsigned 
    
    return h

def hex2int(h_string):
    ''' Function to convert hex strings to 32 bit integers
    '''
    # Convert from hex to Python int
    i = int(h_string, 16)
    
    # Makes Python int to a C integer
    cp = pointer(c_int32(i))
    
    # Type cast int pointer to a float pointer
    fp = cast(cp, POINTER(c_int32))
    
    # Dereference the pointer to get the float
    return fp.contents.value

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
        
    def call_command(self):
        pass

if __name__ == "__main__":
    pass