from grpy.mb80v2 import MB80v2
import time
# Start MAVLink interface
print("Starting...")
mb = MB80v2(sim=False, verbose=False, log=False)

mb.setRetry('_UPST_ADDRESS', 105) # Set computer upstream IP address to 192.168.168.x
mb.setRetry('UPST_LOOP_DELAY', 4) # Set upstream main TX rate (1000/freqHz)

while True:
    data = mb.get()
    """
    dict_keys(['param_value', 'debug_timings', 'voltage', 'twist_linear', 'imu_angular_velocity', 'imu_euler', 'version', 'swing_mode', 'z_rel', 'joint_cmd', 'user', 'joy_buttons', 'debug_legH', 'diagnostics', 'se2twist_des', 'joint_velocity', 'joint_residual', 'joint_status', 'joy_twist', 'behavior', 'slope_est', 'joint_position', 'contacts', 'joint_current', 'y', 'mode', 'joint_temperature', 'joint_voltage', 'imu_linear_acceleration', 'phase', 't'])
    """
    # print("Vel: {}".format(data["twist_linear"]))
    print(data["imu_euler"][1])
    # print("Joint pos: {}".format(data["joint_position"]))
    time.sleep(0.1)

