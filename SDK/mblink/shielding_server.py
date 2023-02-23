import socket
import time
import numpy as np
import struct
from grpy.mb80v2 import MB80v2
from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from safety_enforcer import SafetyEnforcer
import os
import sys
import select
import pickle

safetyEnforcer = SafetyEnforcer(parent_dir=os.getcwd(), epsilon=0.03)

server_socket = socket.socket()
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
HOST = '192.168.168.105'  # Standard loopback interface address (localhost)
PORT = 65495        # Port to listen on (non-privileged ports are > 1023)
thread_count = 0

control_node_on = False
serial_node_on = False
vicon_node_on = False
clients = dict()

try:
    server_socket.bind((HOST, PORT))
except socket.error as e:
    print(str(e))

print('Socket is listening..')
server_socket.listen(5)  

while not control_node_on or not serial_node_on or not vicon_node_on:
    client, address = server_socket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    client_type = client.recv(1024)
    if client_type.decode('utf-8') == "Control":
        print("Control Node attached")
        clients["control"] = client
        control_node_on = True
    elif client_type.decode('utf-8') == "Vicon":
        print("Vicon Node attached")
        clients["vicon"] = client
        vicon_node_on = True
    elif client_type.decode('utf-8') == "Serial":
        print("Serial Node attached")
        clients["serial"] = client
        serial_node_on = True

    thread_count += 1
    print('Thread Number: ' + str(thread_count))

print("Turn on robot mblink...")
# Start MAVLink interface
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

current_stance = np.zeros(12)

try:
    cur_time = time.time()
    dt = 1./250.
    controller_forward = InverseKinematicsController(dt=dt, L=1.0, T=0.08)
    controller_forward_slow = InverseKinematicsController(dt=dt, L=1.0, T=0.15)
    controller_forward_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, Lrot=-0.4)
    controller_forward_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, Lrot=0.4)
    controller_backward = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0)
    controller_backward_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0, Lrot=-0.4)
    controller_backward_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=0, Lrot=0.4)
    controller_lateral_left = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=-90)
    controller_lateral_right = InverseKinematicsController(dt=dt, L=1.0, T=0.08, angle=90)

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

except KeyboardInterrupt:
    sittingDown()
    mb.rxstop()

state = np.zeros(36)

timestamp = []
state_array = []
action_array = []
shielding_status = []
command_status = []

received_vicon = False
received_serial = False

while True:
    try:
        ready_command = select.select([clients["control"]], [], [], 0.01)
        if ready_command[0]:
            control_data = clients["control"].recv(1024)
            data = control_data.decode("utf-8")
            # print(data)
            
        ready_vicon = select.select([clients["vicon"]], [], [], 0.01)
        if ready_vicon[0]:
            vicon_struct = clients["vicon"].recv(1024)
            try:
                vicon_data = struct.unpack("fff", bytearray(vicon_struct))
                # print("{:.3f}, {:.3f}, {:.3f}".format(vicon_data[0], vicon_data[1], vicon_data[2]))
                state[0] = vicon_data[0] # x
                state[1] = vicon_data[1] # y
                state[2] = vicon_data[2] # z

                received_vicon = True
            except Exception as e:
                pass

        ready_serial = select.select([clients["serial"]], [], [], 0.01)
        if ready_serial[0]:
            serial_struct = clients["serial"].recv(1024)
            try:
                serial_data = struct.unpack("f"*33, bytearray(serial_struct))
                # print(serial_data)
                # print("{:.3f}, {:.3f}, {:.3f}".format(serial_data[0], serial_data[1], serial_data[2]))
                
                # map vel
                state[3:6] = np.array(serial_data[15:18]).astype(np.float)
                
                # map imu x, y, z
                state[6:9] = np.array(serial_data[0:3]).astype(np.float)

                # map ang vel
                state[9:12] = np.array(serial_data[18:21]).astype(np.float)
                
                # map joint position
                state[12:24] = np.array(serial_data[3:15]).astype(np.float).reshape((4, 3))[:, [1, 0, 2]].reshape(-1)
                
                # map joint velocity
                state[24:36] = np.array(serial_data[21:33]).astype(np.float).reshape((4, 3))[:, [1, 0, 2]].reshape(-1)
                
                received_serial = True
            except Exception as e:
                pass
        
        if time.time() - cur_time > dt:
            if data == "0" or data == "5":
                action = stable_stance
            else:
                if data == "8":
                    action = controller_forward_slow.get_action().reshape((4, 3))
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
                elif data == "8s":
                    if received_serial and received_vicon:
                        spirit_joint_pos = state[12:24]

                        action = controller_forward.get_action()
                        ctrl = action - spirit_joint_pos
                        ctrl = safetyEnforcer.get_action(state, ctrl) # THIS IS JOINT POS INCREMENT
                        action = (ctrl + spirit_joint_pos).reshape((4, 3))

                        # clipped_action = []
                            
                        # abduction_increment_max = 2.0
                        # abduction_increment_min = -2.0
                        # hip_increment_max = 2.0
                        # hip_increment_min = -2.0
                        # knee_increment_max = 2.0
                        # knee_increment_min = -2.0
                        
                        # abduction_min = -0.5
                        # abduction_max = 0.5
                        # hip_min = 0.5
                        # hip_max = 2.64
                        # knee_min = 0.5
                        # knee_max = 2.64

                        # for i, j in enumerate(ctrl):
                        #     if i % 3 == 0:
                        #         clipped_action.append(
                        #             np.clip(
                        #                 spirit_joint_pos[i] + np.clip(
                        #                     j, abduction_increment_min, abduction_increment_max
                        #                 ), 
                        #                 abduction_min, abduction_max
                        #             )
                        #         )
                        #     elif i % 3 == 1:
                        #         clipped_action.append(
                        #             np.clip(
                        #                 spirit_joint_pos[i] + np.clip(
                        #                     j, hip_increment_min, hip_increment_max
                        #                 ), 
                        #                 hip_min, hip_max
                        #             )
                        #         )
                        #     elif i % 3 == 2:
                        #         clipped_action.append(
                        #             np.clip(
                        #                 spirit_joint_pos[i] + np.clip(
                        #                     j, knee_increment_min, knee_increment_max
                        #                 ), 
                        #                 knee_min, knee_max
                        #             )
                        #         )

                        # action = np.array(clipped_action).reshape((4, 3))

                        received_vicon = False
                        received_serial = False
                try:
                    if action.ndim > 1:
                        action[:, [0, 1]] = action[:, [1, 0]]
                        action = action.reshape(-1)
                except:
                    action = stable_stance
            cur_time = time.time()

        limbCmd(action)
        
        timestamp.append(cur_time)
        state_array.append(state.copy())
        action_array.append(action)
        shielding_status.append(safetyEnforcer.is_shielded)
        command_status.append(data)

    except KeyboardInterrupt:
        sittingDown()
        mb.rxstop()
        
        with open('data-{}-{}.pkl'.format(int(time.time()), safetyEnforcer.epsilon), 'wb') as file:
            pickle.dump({
                "time": timestamp,
                "state": state_array,
                "action": action_array,
                "is_shielded": shielding_status,
                "command": command_status
            }, file)
        break