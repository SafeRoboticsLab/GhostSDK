import rospy
from std_msgs.msg import UInt32
from geometry_msgs.msg import Twist, Pose
import time

# init node
rospy.init_node("gazebo_example", anonymous=True)

# make some subscribers
mode_pub = rospy.Publisher("/vision60/behaviorMode", UInt32, queue_size=1)
id_pub = rospy.Publisher("/vision60/behaviorId", UInt32, queue_size=1)
pose_pub = rospy.Publisher("/vision60/pose", Pose, queue_size=1)
twist_pub = rospy.Publisher("/vision60/twist", Twist, queue_size=1)

print("NOTE: If it looks like this script isn't working, double check the namespacing of the topics created here and running in your simulator.")

# instantiate all the messages we'll use
msg = UInt32()
pose_msg = Pose()
twist_msg = Twist()


# set mode = 0, just for fun, this shouldn't do anything
print("set mode = 0")
msg.data = 0
mode_pub.publish(msg)
time.sleep(0.5)

# set mode = 1
# this should make the robot stand, but it won't
# be in look around mode
print("set mode = 1")
msg.data = 1
mode_pub.publish(msg)
time.sleep(10)


# set id = 1, this is the signal for moving to look around
# mode, but the behaviorId can only be changed when 
# behaviorMode = 0
print("set id = 1")
msg.data = 1
id_pub.publish(msg)
time.sleep(2)

# set mode = 0, so that the id transition can happen
# robot should now officially be in look around mode
print("set mode = 0")
msg.data = 0
mode_pub.publish(msg)
time.sleep(1)

# let's look that way
pose_msg.orientation.x = 0.15
pose_msg.orientation.y = 0.23
pose_msg.orientation.z = 0.15
pose_msg.orientation.w = 0.95
pose_msg.position.z = 1.0
pose_pub.publish(pose_msg)
time.sleep(3)


# now let's look straight again
pose_msg.orientation.x = 0
pose_msg.orientation.y = 0
pose_msg.orientation.z = 0
pose_msg.orientation.w = 0
pose_msg.position.z = 1.0
pose_pub.publish(pose_msg)
time.sleep(3)


# set mode = 1, this should put the robot in walk mode.
# to recap mode = 1 & id = 1 --> walk
print("set mode = 1")
msg.data = 1
mode_pub.publish(msg)
time.sleep(1)

# let's walk this way
twist_msg.linear.x = 0.5
twist_msg.angular.z = -0.25
twist_pub.publish(twist_msg)
time.sleep(5)

# now let's walk that way
twist_msg.linear.x = 0
twist_msg.linear.y = 0.5
twist_msg.angular.z = 0
twist_pub.publish(twist_msg)
time.sleep(5)

# probably time to stop walking
twist_msg.linear.y = 0
twist_pub.publish(twist_msg)
time.sleep(3)

# set mode = 0, this should put us back in look around mode
print("set mode = 0")
msg.data = 0
mode_pub.publish(msg)
time.sleep(5)

# set id = 0, should sit the robot down
print("set id = 0")
msg.data = 0
id_pub.publish(msg)
time.sleep(2)