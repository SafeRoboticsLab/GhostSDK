import sys
import rospy
from gameplay_rollout.srv import *
import numpy as np
import socket
import pickle
import select

rospy.wait_for_service('gameplay_rollout')
gamplay_rollout = rospy.ServiceProxy('gameplay_rollout', GameplayRollout)

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495           # The port used by the server

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(bytes(b'Gameplay'.encode('utf-8')))
    isConnected = True
except OSError:
    try:
        s.sendall(bytes(b'Gameplay'.encode('utf-8')))
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)
horizon = 50

while True:
    try:
        ready = select.select([s], [], [], 0.01)
        if ready[0]:
            received_data = s.recv(1024)
            data = pickle.loads(received_data)
            state = data[0]
            action = data[1]
            res = gamplay_rollout(state, action, horizon)
            s.send(pickle.dumps([res.state, res.done, res.g_x, res.l_x]))

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    except Exception as e:
        print(e)
        pass
