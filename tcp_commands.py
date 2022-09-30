''' PyAutoSTM

    File containing functions to do TCP communications via the Nanonis v5 TCP protocol
    
    Juwan Jeremy Jacobe
    University of Notre Dame
    
    Last Updated: 30 Sept 2022
'''

import numpy as np
import socket
import bitstring
import struct
from ctypes import *

## Helper Functions
def append_command(command, len_max):
    new_command = command
    
    while (len(new_command) < len_max):
        new_command = new_command + b'\x00'
        
    return new_command

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
 
# Stuff from George's example
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

# Stuff from George's example
def hex2int(s):
    ''' Function to convert hex strings to 32 bit integers
    '''
    
    h_string = str(s).hex()
    
    # Convert from hex to Python int
    i = int(h_string, 16)
    
    # Makes Python int to a C integer
    cp = pointer(c_int32(i))
    
    # Type cast int pointer to a float pointer
    fp = cast(cp, POINTER(c_int32))
    
    # Dereference the pointer to get the float
    return fp.contents.value

def create_header(name, body_size):
    header = append_command(name, 32)
    
    header = header + int2byte(body_size, size = 32)
    header = header + int2byte(1, size = 16) # set response to true
    header = header + int2byte(0, size = 16) # filler
    
    return header

## Commands proper
# Note, every integer/float/double byte representation is in big-endian representation
# Implementing Scan.FrameDataGrab in this new framework
class ScanData():
    def __init__(self):
        data = np.array([])
        channels = 0
        lines = 0
        pixels = 0

def get_scan_buffer(client):
    name = b'Scan.BufferGet'
    
    header = create_header(name, body_size = 4)
    
    message = header
    
    client.sock.send(message)
    
    reply = client.sock.recv(1024)
    
    # Converting number of channels, number of pixels, and lines to 
    num_channels = hex2double(reply[40:44])
    pixels = hex2double(reply[44 + num_channels*4: 48 + num_channels*4])
    lines = hex2double(reply[48 + num_channels*4: 52 + num_channels*4])
    
    return num_channels, pixels, lines
    
def scan_frame_grab(client, chan, direc, lines, pixels, send_response = 1):
    ''' Function to grab the data of a scan frame
    
        Args:
            chan (4 byte): channel to read data from, default is z?
            direc (4 byte): 0 or 1, forwards or backwards
            lines (int): argument received from get_scan_buffer, to calculate size of data block
            pixels (int): argument received from get_scan_buffer, to calculate size of data block
    '''
    name = b'Scan.FrameDataGrab'
    header = create_header(name, body_size = 8)
    
    body = int2byte(chan) + int2byte(direc)
    
    message = header + body
    
    client.sock.send(message)
    
    # Calculate size of data block and read reply
    data_size = 4*lines*pixels
    reply = client.sock.recv(data_size + 1024)
    
    # Grab size of body and channel name string -- need this to calculat4e position where scan data block starts
    body_size = hex2int(reply[32:36])
    name_size = hex2int(reply[40:44])
    
    # Making sure whole message was gotten :>
    while len(reply) < body_size:
        reply += sock_data.recv(data_size)
        
    data_array = np.frombuffer(reply[52 + name_size:52+name_size+data_size], dtype = np.float32)
    
    # convert from little to big endian using dtype = '>f4'
    data = np.ndarray(shape = (lines, pixels), dtype = '>f4', buffer = data_array)
    
    return data