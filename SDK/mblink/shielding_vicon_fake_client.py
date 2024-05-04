import socket
import socket
import struct
import time

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495  # The port used by the server

isConnected = False
goal = [0, 0]

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes(b'Vicon'.encode('utf-8')))
    isConnected = True
except OSError:
    try:
        s.sendall(bytes(b'Vicon'.encode('utf-8')))
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)

while isConnected:
    vicon_data = struct.pack("!9f", 0, 0, 0, 0, 0, 0, 0, goal[0], goal[1])

    try:
        s.sendall(vicon_data)
    except Exception as e:
        print(e)
        time.sleep(0.001)
