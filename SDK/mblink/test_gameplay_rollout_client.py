import sys
import rospy
from gameplay_rollout.srv import *
import time
import numpy as np

def gameplay_rollout(state, action, horizon):
    rospy.wait_for_service('gameplay_rollout')
    try:
        gamplay_rollout = rospy.ServiceProxy('gameplay_rollout', GameplayRollout)
        res = gamplay_rollout(state, action, horizon)
        return res.state, res.done, res.g_x, res.l_x
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    # state = [0.35, 0.1, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.75, 1.45, 0.5, 0.75, 1.45, -0.5, 0.75, 1.45, -0.5, 0.75, 1.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # action = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    horizon = 10
    print("Requesting gameplay rollout")
    cur_time = time.time()
    for i in range(1000):
        state = list(np.random.uniform(-1.0, 1.0, 33))
        action = list(np.random.uniform(-0.5, 0.5, 12))
        gameplay_rollout(state, action, horizon)
    print((time.time() - cur_time)/1000.0)
