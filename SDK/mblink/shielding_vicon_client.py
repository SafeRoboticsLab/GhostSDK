import time
import socket
import rospy
from geometry_msgs.msg import TransformStamped
import socket
import struct

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495           # The port used by the server

isConnected = False

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

def callback(data):
    # rospy.loginfo("{:.3f}, {:.3f}, {:.3f}".format(data.transform.translation.x, data.transform.translation.y, data.transform.translation.z))
    vicon_data = struct.pack(
        "!7f", 
        data.transform.translation.x, 
        data.transform.translation.y, 
        data.transform.translation.z - 0.04, 
        data.transform.rotation.x, 
        data.transform.rotation.y, 
        data.transform.rotation.z, 
        data.transform.rotation.w
    )

    s.sendall(vicon_data)
    
def listener():
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber("/vicon/spirit/spirit", TransformStamped, callback)
    rospy.spin()

if isConnected:
    listener()