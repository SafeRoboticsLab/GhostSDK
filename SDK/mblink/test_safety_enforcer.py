from safety_enforcer import SafetyEnforcer
import numpy as np
import time

for i in range(8):
    print("Test version {}".format(i))
    safetyEnforcer = SafetyEnforcer(version=i)

    start_time = time.time()
    safetyEnforcer.get_action(np.zeros(36), np.zeros(12))
    print(time.time() - start_time)

    start_time = time.time()
    for i in range(100):
        safetyEnforcer.get_action(np.zeros(36), np.zeros(12))
    print((time.time() - start_time) * 0.01)

    start_time = time.time()
    for i in range(1000):
        safetyEnforcer.get_action(np.zeros(36), np.zeros(12))
    print((time.time() - start_time) * 0.001)
