from naive_safety_enforcer import NaiveSafetyEnforcer
import numpy as np

for i in range(4):
    print("Version", i)
    safetyEnforcer = NaiveSafetyEnforcer(version=i)
    safetyEnforcer.get_action(np.zeros(36), np.zeros(12))
