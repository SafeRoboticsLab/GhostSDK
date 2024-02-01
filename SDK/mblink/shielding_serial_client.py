import serial
import socket
import struct
import numpy as np
import time

port = "/dev/ttyUSB0"
baud = 115200
ser = serial.Serial(port, baud)

serial_is_on = False

if not ser.isOpen():
    ser.open()

while not serial_is_on:
    try:
        # imu x, y, z, 
        # hip0, knee0, abduction0, 
        # hip1, knee1, abduction1, 
        # hip2, knee2, abduction2, 
        # hip3, knee3, abduction3, 
        # velx, vely, velz
        
        data = ser.readline()
        state_array = data.decode('utf-8').replace('\r\n', '').split(",")
        if len(state_array) == 21:
            serial_is_on = True
    
    except Exception as e:
        print("WARNING: no serial state: {}".format(e))
        serial_is_on = False
        continue

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495           # The port used by the server

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes('Serial', 'utf-8'))
    isConnected = True
except OSError:
    try:
        s.sendall(bytes('Serial', 'utf-8'))
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)
last_time = time.time()
old_joint_pos = np.zeros(12)

while ser.isOpen():
    try:
        """
		roll, pitch, yaw,
		[hip, knee, abduction] @ [0, 1, 2, 3],
		lin_accel_x, lin_accel_y, lin_accel_z,
		ang_vel_x, ang_vel_y, ang_vel_z
		"""
        
        data = ser.readline()
        state_array = np.array(data.decode('utf-8').replace('\r\n', '').split(",")).astype(np.float32)
        if len(state_array) == 21:
            state_array[15:18] = state_array[15:18] * (time.time() - last_time) # accel to vel
            state_array = np.concatenate((state_array, (state_array[3:15] - old_joint_pos) / (time.time() - last_time)), axis=0) # joint vel
            old_joint_pos = state_array[3:15]

            serial_data = struct.pack(
                "!33f", 
                *state_array
            )
            
            s.sendall(serial_data)
            last_time = time.time()

    except Exception as e:
        print("WARNING: no serial state: {}".format(e))
        continue

