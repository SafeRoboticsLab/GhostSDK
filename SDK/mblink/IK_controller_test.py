from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
import time, sys
import numpy as np
from grpy.mb80v2 import MB80v2

# Start MAVLink interface
print("Starting...")
mb = MB80v2(sim=False, verbose=True, log=False)

mb.setRetry('_UPST_ADDRESS', 105) # Set computer upstream IP address to 192.168.168.x
mb.setRetry('UPST_LOOP_DELAY', 4) # Set upstream main TX rate (1000/freqHz)

def limbCmd(pos):
    kp = [80, 50, 80]
    # kd = [5, 2, 5]
    kd = [4, 2, 4]
    # Received om the mb in order (pos[3], Kp[3], Kd[3])x4
    data = np.zeros(58)

    # Populate with the provided data and send
    singleLimb = lambda pos3 : np.hstack((pos3, kp, kd))
    data[:36] = np.hstack([singleLimb(pos[3*i:3*i+3]) for i in range(4)])

    mb.sendUser(data)

def sitting():
    limbCmd(
    [
        0.2, -0.1, 0.2,
        0.2, -0.1, 0.2,
        0.2, 0.1, 0.2,
        0.2, 0.1, 0.2
    ])

def standingUp():
    traj_hip = np.linspace(0.2, 0.9, 100)
    traj_knee = np.linspace(0.2, 1.6, 100)
    target_abduction = -0.07

    traj_idx = 0
    cur_time = time.time()

    while True:
        try:
            res = mb.get()

            target_hip = traj_hip[traj_idx]
            target_knee = traj_knee[traj_idx]
            
            limbCmd(
                [
                    target_hip, target_abduction, target_knee,
                    target_hip, target_abduction, target_knee,
                    target_hip, -1.0 * target_abduction, target_knee,
                    target_hip, -1.0 * target_abduction, target_knee
                ])
            
            if time.time() - cur_time >  0.02:
                if traj_idx < len(traj_hip)-1:
                    traj_idx += 1
                    cur_time = time.time()
                else:
                    break

        except KeyboardInterrupt:
            mb.rxstop()
            break

def sittingDown():
    traj_hip = np.linspace(0.9, 0.2, 100)
    traj_knee = np.linspace(1.6, 0.2, 100)
    target_abduction = -0.07

    traj_idx = 0
    cur_time = time.time()

    while True:
        try:
            res = mb.get()

            target_hip = traj_hip[traj_idx]
            target_knee = traj_knee[traj_idx]
            
            # hip, abduction, knee
            limbCmd(
                [
                    target_hip, target_abduction, target_knee,
                    target_hip, target_abduction, target_knee,
                    target_hip, -1.0 * target_abduction, target_knee,
                    target_hip, -1.0 * target_abduction, target_knee
                ])
            
            if time.time() - cur_time >  0.02:
                if traj_idx < len(traj_hip)-1:
                    traj_idx += 1
                    cur_time = time.time()
                else:
                    break

        except KeyboardInterrupt:
            mb.rxstop()
            break

def main():
    current_stance = np.zeros(12)

    try:        
        cur_time = time.time()
        
        dt = 1./250.
        # 1. really slow walking
        # controller = InverseKinematicsController(dt=dt, L=0.8, T=1.0)
        
        # 2. faster walking
        # 2.1. different phase shift for each leg
        # controller = InverseKinematicsController(dt=dt, offset=[0.0, 0.75, 0.25, 0.5], L=0.8, T=0.1)
       
        ## 2.2. trotting gait
        controller = InverseKinematicsController(dt=dt, L=0.8, T=0.1)

        while time.time() - cur_time < 2:
            sitting()
        
        current_stance = np.array([
            0.2, -0.1, 0.2,
            0.2, -0.1, 0.2,
            0.2, 0.1, 0.2,
            0.2, 0.1, 0.2
        ])

        standingUp()

        current_stance = np.array([
            0.9, -0.07, 1.6,
            0.9, -0.07, 1.6,
            0.9, 0.07, 1.6,
            0.9, 0.07, 1.6
        ])

        # hip, abduction, knee
        action = controller.get_action().reshape((4, 3)) # [FL, BL, FR, BR] x [abduction, hip, knee]
        action[:, [0, 1]] = action[:, [1, 0]]
        action = action.reshape(-1)

        # bridge the current stance with the initial stance of IK controller
        stand_to_walking_traj = np.linspace(current_stance, action, 50)
        cur_time = time.time()
        traj_idx = 0
        
        while True:
            state = stand_to_walking_traj[traj_idx]
            limbCmd(state)

            if time.time() - cur_time >  0.02:
                if traj_idx < len(stand_to_walking_traj) - 1:
                    traj_idx += 1
                    cur_time = time.time()
                else:
                    break

        while True:
            if time.time() - cur_time > dt:
                action = controller.get_action().reshape((4, 3))
                action[:, [0, 1]] = action[:, [1, 0]]
                action = action.reshape(-1)
                cur_time = time.time()
            
            limbCmd(action)

    except KeyboardInterrupt:
        sittingDown()
        mb.rxstop()

if __name__ == "__main__":
    main()