import serial
import numpy as np

ser = serial.Serial('/dev/ttyUSB0', 115200)
if not ser.isOpen():
	ser.open()

while True:
	try:
		"""
		roll, pitch, yaw,
		[hip, knee, abduction] @ [0, 1, 2, 3],
		lin_accel_x, lin_accel_y, lin_accel_z,
		ang_vel_x, ang_vel_y, ang_vel_z
		""" 
		data = ser.readline()
		state_array = data.decode('utf-8').replace('\r\n', '').split(",")
		if len(state_array) == 21:
			state = np.array(state_array).astype(np.float)
			print("{:.3f}\t{:.3f}\t{:.3f}".format(state[15], state[16], state[17]))
	except Exception as e:
		continue