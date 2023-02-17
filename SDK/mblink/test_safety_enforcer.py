from safety_enforcer import SafetyEnforcer
import numpy as np
import time

# state:
# obs = x, y, z, roll (imu.euler.x), pitch (imu.euler.y), yaw (imu.euler.z), velx, vely, velz
# state = np.concatenate((spirit_new_obs, spirit_old_obs, spirit_new_joint_pos, spirit_old_joint_pos), axis=0)

safetyEnforcer = SafetyEnforcer()

start_time = time.time()
safetyEnforcer.get_action(np.zeros(42), np.zeros(12))
print(time.time() - start_time)

start_time = time.time()
for i in range(100):
    safetyEnforcer.get_action(np.zeros(42), np.zeros(12))
print((time.time() - start_time) * 0.01)

start_time = time.time()
for i in range(1000):
    safetyEnforcer.get_action(np.zeros(42), np.zeros(12))
print((time.time() - start_time) * 0.001)