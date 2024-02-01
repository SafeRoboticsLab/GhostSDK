from naive_safety_enforcer import NaiveSafetyEnforcer
import numpy as np

safetyEnforcer = NaiveSafetyEnforcer(version=0)
safetyEnforcer.get_action(np.zeros(36), np.zeros(12))

safetyEnforcer = NaiveSafetyEnforcer(version=1)
safetyEnforcer.get_action(np.zeros(36), np.zeros(12))