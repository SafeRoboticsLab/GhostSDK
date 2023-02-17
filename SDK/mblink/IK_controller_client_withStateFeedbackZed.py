from inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
import time, sys
import numpy as np
from grpy.mb80v2 import MB80v2
import socket
import select
import serial
import pyzed.sl as sl
import threading
from safety_enforcer import SafetyEnforcer
import os

safetyEnforcer = SafetyEnforcer(parent_dir=os.getcwd(), epsilon=-0.01)

# intialize ZED cam
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
# Use a right-handed Y-up coordinate system
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
init_params.coordinate_units = sl.UNIT.METER  # Set units in meters

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Enable positional tracking with default parameters
py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
tracking_parameters = sl.PositionalTrackingParameters(
    _init_pos=py_transform
)

initial_position = sl.Transform()
# Set the initial positon of the Camera Frame at 1m80 above the World Frame
initial_translation = sl.Translation()
initial_translation.init_vector(0.277, 0.06, 0.15)
initial_position.set_translation(initial_translation)
tracking_parameters.set_initial_world_transform(initial_position)

err = zed.enable_positional_tracking(tracking_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Track the camera position during 1000 frames
zed_pose = sl.Pose()
runtime_parameters = sl.RuntimeParameters()

while not zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    pass

# initialize serial port
state = [
    np.zeros(9), # spirit_new_obs
    np.zeros(9), # spirit_old_obs
    np.zeros(12),# spirit_new_joint_pos
    np.zeros(12),# spirit_old_joint_pos
]

is_initialized = False

def get_state(port, baud):
    global state
    global is_initialized

    serial_is_on = False
    zed_is_on = False

    ser = serial.Serial(port, baud)
    if not ser.isOpen():
        ser.open()

    while True:
        try:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get the pose of the left eye of the camera with reference to the world frame
                zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

                # Display the translation and timestamp
                py_translation = sl.Translation()
                tx = round(zed_pose.get_translation(py_translation).get()[0], 3) - 0.277
                ty = round(zed_pose.get_translation(py_translation).get()[1], 3) - 0.06
                tz = round(zed_pose.get_translation(py_translation).get()[2], 3) - 0.15
                state[0][0] = tx
                state[0][1] = ty
                state[0][2] = tz

                zed_is_on = True

        except Exception as e:
            print("WARNING: no ZED pose: {}".format(e))
            zed_is_on = False
            continue

        try:
            # imu x, y, z, 
            # hip0, knee0, abduction0, 
            # hip1, knee1, abduction1, 
            # hip2, knee2, abduction2, 
            # hip3, knee3, abduction3, 
            # velx, vely, velz
            
            data = ser.readline()
            state_array = data.decode('utf-8').replace('\r\n', '').split(",")
            if len(state_array) == 18:
                # map imu x, y, z
                state[0][3:6] = np.array(state_array[:3]).astype(np.float)
                # map joint position
                state[2] = np.array(state_array[3:15]).astype(np.float).reshape((4, 3))[:, [1, 0, 2]].reshape(-1)
                # map vel
                state[0][6:] = np.array(state_array[15:]).astype(np.float)

                serial_is_on = True
        except Exception as e:
            print("WARNING: no serial state: {}".format(e))
            serial_is_on = False
            continue
        
        if serial_is_on and zed_is_on:
            is_initialized = True

try:
   t = threading.Thread(target=get_state, args=('/dev/ttyUSB0', 115200))
   t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
   t.start()
except:
   print("Error: unable to start thread")

while not is_initialized:
    time.sleep(0.01)

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
                    elif data == "s":
                        spirit_joint_pos = state[2]

                        action = controller_forward.get_action()
                        ctrl = action - spirit_joint_pos
                        s_ = np.concatenate(state, axis=0)
                        ctrl = safetyEnforcer.get_action(s_, ctrl) # THIS IS JOINT POS INCREMENT

                        action = (ctrl + spirit_joint_pos).reshape((4, 3))
                        
                        # clip the increment of action from safetyEnforcer and return NEW JOINT POS
                        # clipped_action = []
                        
                        # abduction_increment_max = 0.5
                        # abduction_increment_min = -0.5
                        # hip_increment_max = 0.5
                        # hip_increment_min = -0.5
                        # knee_increment_max = 0.5
                        # knee_increment_min = -0.5
                        
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

                        state[1] = state[0]
                        state[3] = state[2]
                    try:
                        if action.ndim > 1:
                            action[:, [0, 1]] = action[:, [1, 0]]
                            action = action.reshape(-1)
                    except:
                        action = stable_stance
                cur_time = time.time()

            limbCmd(action)

    except KeyboardInterrupt:
        sittingDown()
        mb.rxstop()

if __name__ == "__main__":
    main()