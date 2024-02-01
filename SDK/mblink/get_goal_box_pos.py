# run with python 2
import rospy
from geometry_msgs.msg import TransformStamped
import socket

def callback(data):
    rospy.loginfo("{:.3f}, {:.3f}, {:.3f}".format(data.transform.translation.x, data.transform.translation.y, data.transform.translation.z-0.04))
    
def listener():
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber("/vicon/goal_box/goal_box", TransformStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()