"""
This script should be run on a computer connected to the upstream ethernet port on the mainboard.
Its IP should be set statically to 192.168.168.x. If you would like to receive only on a particular
IP, set _UPST_ADDRESS below accordingly.

This script sends desired toe positions and gains, and these are read by the UserLimbCmd behavior
in LimbControl/main.cpp (which should be flashed on the mainboard)
"""
import time, sys
import numpy as np
from grpy.mb80v2 import MB80v2

# Start MAVLink interface
print("Starting...")
mb = MB80v2(sim=False, verbose=True, log=False)

mb.setRetry('_UPST_ADDRESS', 105) # Set computer upstream IP address to 192.168.168.x
mb.setRetry('UPST_LOOP_DELAY', 4) # Set upstream main TX rate (1000/freqHz)

"""Function to set all limb coord positions"""
def limbCmd(kp, kd, pos):
    # Received om the mb in order (pos[3], Kp[3], Kd[3])x4
    data = np.zeros(58)

    # Populate with the provided data and send
    singleLimb = lambda pos3 : np.hstack((pos3, kp, kd))
    data[:36] = np.hstack([singleLimb(pos[3*i:3*i+3]) for i in range(4)])

    mb.sendUser(data)

# Start
cur_time = time.time()
abduction_range = [-0.7, 0.7]
hip_range = [0.0, 1.5]
knee_range = [0.0, 2.5]

# sitting
# target_abduction = 0.0
# target_hip = 0.2
# target_knee = 0.2

# slightly off the ground
# target_abduction = 0.0
# target_hip = 0.4
# target_knee = 0.7

sample_traj_hip = np.concatenate((np.linspace(0.2, 0.7, 100), np.linspace(0.7, 0.2, 100)))
sample_traj_knee = np.concatenate((np.linspace(0.2, 1.4, 100), np.linspace(1.4, 0.2, 100)))

traj_idx = 0

target_abduction = -0.1
target_hip = sample_traj_hip[traj_idx]
target_knee = sample_traj_knee[traj_idx]

def sitting():
    limbCmd(
    [150, 50, 150], 
    [5, 2, 5], 
    [
        0.2, -0.1, 0.2,
        0.2, -0.1, 0.2,
        0.2, 0.1, 0.2,
        0.2, 0.1, 0.2
    ])

while time.time() - cur_time < 2:
    sitting()

cur_time = time.time()

while True:
    try:
        # This blocks till there is new data in the queue. The rate can be set (to a limit) by UPST_LOOP_DELAY = 1000/freqHz
        res = mb.get()
        # hip, abduction, knee
        limbCmd(
            [150, 50, 150], 
            [5, 2, 5], 
            [
                target_hip, target_abduction, target_knee,
                target_hip, target_abduction, target_knee,
                target_hip, -1.0 * target_abduction, target_knee,
                target_hip, -1.0 * target_abduction, target_knee
            ])
        
        if time.time() - cur_time >  0.05:
            traj_idx = (traj_idx + 1) % len(sample_traj_hip)
            target_hip = sample_traj_hip[traj_idx]
            target_knee = sample_traj_knee[traj_idx]
            cur_time = time.time()

    except KeyboardInterrupt:
        mb.rxstop()
        break