# run with python 2
import rospy
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion
import socket
import numpy as np

def map_rad_range(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def calculate_angle(point_a, point_b):
    vector_ab = np.array(point_b) - np.array(point_a)
    angle_rad = map_rad_range(np.arctan2(vector_ab[1], vector_ab[0]) + np.pi/2)
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg

def callback(data):
    # rospy.loginfo("{:.3f}, {:.3f}, {:.3f}".format(data.transform.translation.x, data.transform.translation.y, data.transform.translation.z))
    translation = data.transform.translation
    rotation = data.transform.rotation
    euler_angles = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])
    # print("yaw: {:.3f}".format(euler_angles[2]))
    # print("{:.3f}, {:.3f}, {:.3f}".format(translation.x, translation.y, euler_angles[2]))

    goal = [3.0, 8.0]
    pos = [translation.x, translation.y]
    angle_rad, angle_deg = calculate_angle(pos, goal)
    
    print(np.degrees(map_rad_range(euler_angles[2] - angle_rad + np.pi)))

def listener():
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber("/vicon/spirit/spirit", TransformStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()