''' PyAutoSTM

    File containing functions to do TCP communications via the Nanonis v5 TCP protocol
    
    Juwan Jeremy Jacobe
    University of Notre Dame
    
    Last Updated: 14 Oct 2022
'''

import numpy as np
import bitstring
import struct
from ctypes import *

## Helper Functions -- need to test them to make sure they work as intended
def append_command(command, len_max):
    ''' Function for appending byte string with empty bytes (b'\x00') until byte string
    reaches desired length
    '''
    
    new_command = command
    
    # Append with 0's until len_max is reached
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
    
    return f_val[0]

# double to hex
def double2hex(d_val):
    h = struct.pack('>d', d_val)
    
    return h
    
# hex to double
def hex2double(hex_string):
    d_val = struct.unpack('>d', hex_string)
    
    return d_val[0]

# unsigned int(32) to hex
def unsignedint2hex(I_val):
    h = struct.pack('>I', I_val)
    
    return h
  
# hex to unsigned int(32)
def hex2unsignedint(hex_string):
    I_val = struct.unpack('>I', hex_string)
    
    return I_val[0]

# unsigned int(16) to hex
def unsignedshort2hex(H_val):
    h = struct.pack('>H', H_val)
    
    return h
    
# hex to unsigned int (16)
def hex2unsignedshort(hex_string):
    H_val = struct.unpack('>H', hex_string)
    
    return H_val[0]
    
# string to hex
def str2hex(string):
    string = string.encode('utf-8')
    h = struct.pack('{}s'.format(len(string)), string)

    return h

# hex to string
def hex2str(hex_string):
    string_val = struct.unpack('{}s'.format(len(hex_string)), hex_string)

    return string_val[0].decode('utf-8')

# hex to integer
def hex2integer(hex_string):
    int_val = struct.unpack('>i', hex_string)

    return int_val[0]

# integer to hex
def integer2hex(integer):
    h = struct.pack('>i', integer)

    return h

## Do we need this for what we need so far?
# hex to 1D array 
# 1D array to hex
# hex to 2D array
# 2D array to hex
 
## Stuff from George's example
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

def hex2int(s):
    ''' Function to convert hex strings to 32 bit integers
    '''
    
    h_string = str(s.hex())
    
    # Convert from hex to Python int
    i = int(h_string, 16)
    
    # Makes Python int to a C integer
    cp = pointer(c_int32(i))
    
    # Type cast int pointer to a float pointer
    fp = cast(cp, POINTER(c_int32))
    
    # Dereference the pointer to get the float
    return fp.contents.value

def create_header(name, body_size):
    ''' Command for constructing a header for sending a TCP request message
    
    Args:
        name (hex string): name of the command
        body_size (int): size of the body in bytes
        
    Returns
        header (hex string): the appended hex string to be used as a header
    '''
    
    header = append_command(name, 32)
    
    header = header + int2byte(body_size, size = 32)
    header = header + int2byte(1, size = 16) # set response to true
    header = header + int2byte(0, size = 16) # filler
    
    return header

## Commands proper
# Note, every integer/float/double byte representation is in big-endian representation

### Bias
# Bias.Set
def bias_set(client, bias_val):
    ''' Command for setting bias voltage

    Args:
        client (Nanonis object)
        bias_val (V): bias voltage in units of V
    '''
    
    name = b'Bias.Set'
    
    header = create_header(name, body_size = 4)
    body = float2hex(bias_val)
    
    message = header + body
    
    client.sock.send(message)
    reply = client.sock.recv(1024)

# Bias.Get
def bias_get(client):
    ''' Command for getting bias voltage
    
    Args:
        client (Nanonis object)
        
    Returns:
        bias_val (V)
    '''
    
    name = b'Bias.Get'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
    # Read body for bias voltage set in Nanonis software
    bias_val = hex2float(reply[40:44])
    
    return bias_val
    
### Current

# Current.Get
def current_get(client):
    ''' Command to get the tunneling current value
    
    Args:
        client (Nanonis object)
        
    Returns:
        current_val (A)
    '''
    
    name = b'Current.Get'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
    # Read body for current value
    current_val = hex2float(reply[40:44])
    
    return current_val
    
### Z-Controller
# ZCtrl.OnOffGet
def zctrl_onoffget(client):
    ''' Command to get whether or not Z-Controller is on or off
    
    Args:
        client (Nanonis object)
        
    Returns:
        zctrl_status: 0 if Off, 1 if On
    '''
    
    name = b'ZCtrl.OnOffGet'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
    # Read body for z-ctrl status
    zctrl_status = hex2unsignedint(reply[40:44])
    
    return zctrl_status

# ZCtrl.OnOffSet
def zctrl_onoffset(client, zctrl_status):
    ''' Command to set ZCtrl off or on
    
    Args:
        client (Nanonis object)
        zctrl_status (unsigned int32): either 0 (set OFF) or 1 (set ON)
    '''
    
    name = b'ZCtrl.OnOffSet'
    
    header = create_header(name, body_size = 4)
    body = unsignedint2hex(zctrl_status)
    message = header + body
    
    client.sock.send(message)
    reply = client.sock.recv(1024)

# ZCtrl.SetpntGet
def zctrl_setpntget(client):
    ''' Command for getting the setpoint of the Z-Controller
    
    Args:
        client (Nanonis object)
    
    Returns:
        setpnt (float): set point in Amps
    '''
    
    name = b'ZCtrl.SetpntGet'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
    # Read reply for the setpoint
    setpnt = hex2float(reply[40:44])
    
    return setpnt

# ZCtrl.SetpntSet
def zctrl_setpntset(client, setpnt):
    ''' Command for setting the setpoint of the Z-Controller

    Args:
        client (Nanonis object)
        setpnt (float): desired set point in Amps
    '''
    
    name = b'ZCtrl.SetpntSet'
    
    header = create_header(name, body_size = 4)
    body = float2hex(setpnt)
    message = header + body
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
# ZCtrl.Home
def zctrl_home(client):
    ''' Command for moving the tip to its home position
    
    Args:
        client (Nanonis object)
    '''
    
    name = b'ZCtrl.Home'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    
# ZCtrl.Withdraw
def zctrl_withdraw(client, wait, timeout):
    ''' Command for switching off the Z-Controller and fully withdrawing the tip
    
    Args:
        client (Nanonis object)
        wait (unsigned int32): indicated if function waits until tip is fully withdrawn (=1) or not (=0)
        timeout (int32): time in ms this function waits. Set to -1 to wait indefinitely
    '''
    
    name = b'ZCtrl.Withdraw'
    
    header = create_header(name, body_size = 8)
    body = unsignedint2hex(wait) + int2byte(timeout)
    message = header + body
    
    client.sock.send(message)
    reply = client.sock.recv(1024)

### Scan
class ScanData():
    def __init__(self):
        data = np.array([])
        channels = 0
        lines = 0
        pixels = 0

# Scan.Action
def scan_action(client, action, direction):
    ''' Command to do a scan action, either start, stop, pause or resume a scan
    
    Args:
        client (Nanonis object)
        action (unsigned int16): sets which action to perform, where 0 means Start, 1 is Stop, 2 is Pause
          3 is Resume
        direction (unsigned int32): if 1, scan is direction is set to up, if 0, direction is down    
    '''
    
    name = b'Scan.Action'
    
    header = create_header(name, body_size = 6)
    body = unsignedshort2hex(action) + unsignedint2hex(direction)
    message = header + body
    
    client.sock.send(message)
    reply = client.sock.recv(1024)

# Scan.StatusGet
def scan_statusget(client):
    ''' Command to see if scan is currently running or not
    
    Args: 
        client (Nanonis object)
        
    Returns:
        scan_status (unsigned int32): if 1 scan is running. If 0, scan is not running
    '''
    
    name = b'Scan.StatusGet'
    
    header = create_header(name, body_size = 0)
    message = header
    
    client.sock.send(message)
    reply = client.sock.recv(1024)
    scan_status = hex2unsignedint(reply[40:44])
    
    return scan_status

# Scan.WaitEndofScan
def scan_waitendofscan(client, timeout):
    ''' Command to wait for an end-of-scan, which only returns when
    an End-of-Scan or timeout occurs (whichever occurs first)

    Args:
        client (Nanonis object)
        timeout (int32): sets how many milliseconds this function waits for
        an End-of-Scan. If -1, waits indefinitely

    Return:
        timeout_status (unsigned int32): if 1, function timed-out. If 0, function
        didn't time out
        file_path_size (unsigned int 32): the number of bytes corresponding to the File path
        string
        file_path (str): the path where the data file was automatically saved (if auto-save was on)
        If no file was saved at the End-of-Scan, it returns an empty path 
    '''

    name = b'Scan.WaitEndOfScan'

    header = create_header(name, body_size = 4)
    body = integer2hex(timeout)
    message = header + body

    client.sock.send(message)
    reply = client.sock.recv(1024)
    timeout_status = hex2unsignedint(reply[40:44])
    file_path_size = hex2unsignedint(reply[44:48])

    if file_path_size > 0:
        file_path = hex2str(reply[48:48+file_path_size])
    else:
        file_path = ''

    return timeout_status, file_path_size, file_path

# Scan.FrameSet
def scan_frameset(client, centX, centY, width, height, angle):
    ''' Command to configure the scan frame parameters

    Args:
        client (Nanonis client)
        centX (float32): x position of scan frame center (m)
        centY (float32): y position of scan frame center (m)
        width (float32): width of scan frame (m)
        height (float32): height of scan frame (m)
        angle (float32): the angle of scan frame, where positive angle means clockwise rotation (degrees)
    '''

    name = b'Scan.FrameSet'

    header = create_header(name, body_size = 20)
    body = float2hex(centX) + float2hex(centY) + float2hex(width) + float2hex(height) + float2hex(angle)
    message = header + body

    client.sock.send(message)
    reply = client.sock.recv(1024)

# Scan.FrameGet
def scan_frameget(client):
    ''' Command to return scan frame parameters

    Args:
        client (Nanonis client)

    Returns:
        centX (float32): x position of scan frame center (m)
        centY (float32): y position of scan frame center (m)
        width (float32): width of scan frame (m)
        height (float32): height of scan frame (m)
        angle (float32): the angle of scan frame, where positive angle means clockwise rotation (degrees)
    '''

    name = b'Scan.FrameGet'

    header = create_header(name, body_size = 0)
    message = header

    client.sock.send(message)
    reply = client.sock.recv(1024)

    centX = hex2float(reply[40:44])
    centY = hex2float(reply[44:48])
    width = hex2float(reply[48:52])
    height = hex2float(reply[52:56])
    angle = hex2float(reply[56:60])

    return centX, centY, width, height, angle

# Scan.BufferSet
def scan_bufferset(client, num_channels, channel_indices, pixels, lines):
    ''' Configure the scan buffer parameters

    Args:
        client (Nanonis.client)
        num_channels (int): the number of recorded channels. Defines the size
        of the channels indexes array
        channel_indices (np.array): 1D array of indices of recorded channels. Indices are
        comprised between 0 and 23 for the 24 signals assigned in the Signals Manager
        pixels (int): number of pixels per line
        lines (int): number of scan lines
    '''

    name = b'Scan.BufferSet'

    array_size = num_channels * 4
    header = create_header(name, body_size = 12 + array_size)
    
    # Go through array and convert to hex string for each element
    channel_indices_hex = b''
    for channel_i in channel_indices:
        channel_indices_hex += integer2hex(channel_i)

    body = integer2hex(num_channels) + channel_indices_hex + integer2hex(pixels) + integer2hex(lines)
    message = header + body

    client.sock.send(message)
    reply = client.sock.recv(1024)

# Scan.BufferGet
def scan_bufferget(client):
    name = b'Scan.BufferGet'
    
    header = create_header(name, body_size = 4)
    message = header
    
    client.sock.send(message)
    
    reply = client.sock.recv(1024)
    
    # Converting number of channels, number of pixels, and lines to 
    num_channels = hex2int(reply[40:44])
    pixels = hex2int(reply[44 + num_channels*4: 48 + num_channels*4])
    lines = hex2int(reply[48 + num_channels*4: 52 + num_channels*4])
    
    return num_channels, pixels, lines
    
# Scan.SpeedSet

# Scan.FrameDataGrab
def scan_framedatagrab(client, chan, direc, lines, pixels, send_response = 1):
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
        reply += client.sock.recv(data_size)
        
    data_array = np.frombuffer(reply[52 + name_size:52+name_size+data_size], dtype = np.float32)
    
    # convert from little to big endian using dtype = '>f4'
    data = np.ndarray(shape = (lines, pixels), dtype = '>f4', buffer = data_array)
    
    return data

## Follow Me