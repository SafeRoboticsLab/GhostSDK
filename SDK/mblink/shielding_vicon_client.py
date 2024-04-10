import time
import socket
import rospy
from geometry_msgs.msg import TransformStamped
import socket
import struct

HOST = '192.168.168.105'  # The server's hostname or IP address
PORT = 65495           # The port used by the server

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

def callback_spirit(data):
    global goal
    # rospy.loginfo("{:.3f}, {:.3f}, {:.3f}".format(data.transform.translation.x, data.transform.translation.y, data.transform.translation.z))
    vicon_data = struct.pack(
        "!9f", 
        data.transform.translation.x, 
        data.transform.translation.y, 
        data.transform.translation.z - 0.04, 
        data.transform.rotation.x, 
        data.transform.rotation.y, 
        data.transform.rotation.z, 
        data.transform.rotation.w,
        goal[0],
        goal[1]
    )

    s.sendall(vicon_data)

def callback_goal(data):
    global goal
    goal = [data.transform.translation.x, data.transform.translation.y]
    
def listener():
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber("/vicon/spirit/spirit", TransformStamped, callback_spirit)
    rospy.Subscriber("/vicon/goal_bottle/goal_bottle", TransformStamped, callback_goal)
    rospy.spin()

if isConnected:
    listener()