from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
import time, sys
import numpy as np
from grpy.mb80v2 import MB80v2
import socket
import select

# SETUP COMMAND RECEIVING SERVER
HOST = "192.168.168.105"
PORT = 65432

isConnected = False

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    isConnected = True
except OSError:
    try:
        s.sendall(b'Test')
        isConnected = True
    except OSError:
        print("Something is wrong")
        isConnected = False
    pass

s.setblocking(0)

# Start MAVLink interface
print("Starting...")
mb = MB80v2(sim=False, verbose=False, log=False)

mb.setRetry('_UPST_ADDRESS', 105) # Set computer upstream IP address to 192.168.168.x
mb.setRetry('UPST_LOOP_DELAY', 4) # Set upstream main TX rate (1000/freqHz)

def limbCmd(pos):
    kp = [60, 60, 60]
    kd = [3, 3, 3]

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
        controller_forward = InverseKinematicsController(dt=dt, L=1.0, T=0.08, Xdist=0.464, Ydist=0.33)
        controller_forward_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, Lrot=-0.4, Xdist=0.464, Ydist=0.33)
        controller_forward_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, Lrot=0.4, Xdist=0.464, Ydist=0.33)
        controller_backward = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0, Xdist=0.464, Ydist=0.33)
        controller_backward_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0, Lrot=-0.4, Xdist=0.464, Ydist=0.33)
        controller_backward_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0, Lrot=0.4, Xdist=0.464, Ydist=0.33)
        controller_lateral_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=-90, Xdist=0.464, Ydist=0.33)
        controller_lateral_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=90, Xdist=0.464, Ydist=0.33)

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

        data = "0"
        stable_stance = current_stance

        while isConnected:
            ready = select.select([s], [], [], 0.01)
            if ready[0]:
                data = s.recv(1024)
                data = data.decode("utf-8")
                print(data)

            if time.time() - cur_time > dt:
                if data == "0" or data == "5":
                    action = stable_stance
                else:
                    if data == "8":
                        action = controller_forward.get_action().reshape((4, 3))
                    elif data == "9":
                        action = controller_forward_right.get_action().reshape((4, 3))
                    elif data == "7":
                        action = controller_forward_left.get_action().reshape((4, 3))
                    elif data == "2":
                        action = controller_backward.get_action().reshape((4, 3))
                    elif data == "1":
                        action = controller_backward_left.get_action().reshape((4, 3))
                    elif data == "3":
                        action = controller_backward_right.get_action().reshape((4, 3))
                    elif data == "4":
                        action = controller_lateral_left.get_action().reshape((4, 3))
                    elif data == "6":
                        action = controller_lateral_right.get_action().reshape((4, 3))
                    try:
                        action[:, [0, 1]] = action[:, [1, 0]]
                        action = action.reshape(-1)
                    except:
                        action = stable_stance
                cur_time = time.time()
            
            # data = mb.get()
            # print("{:.3f}\t{:.3f}\t{:.3f}".format(data["imu_euler"][0], data["imu_euler"][1], data["imu_euler"][2]))
            limbCmd(action)

    except KeyboardInterrupt:
        sittingDown()
        mb.rxstop()

if __name__ == "__main__":
    main()