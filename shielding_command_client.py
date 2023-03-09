import time
import socket
import select

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495           # The port used by the server

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes('Control', 'utf-8'))
    isConnected = True
except OSError:
    try:
        s.sendall(bytes('Control', 'utf-8'))
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)

while isConnected:
    data = input()
    s.sendall(bytes(data, 'utf-8'))