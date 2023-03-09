# EXPERIMENT NOTE
Different run types:
## Run with only performance controller, test deployment
Run the following files:
- IK_controller_server.py
- IK_controller_client.py

Result: Work well

## Run with value shielding and get pose from ZED + IMU
Run the following files:
- IK_controller_server.py
- IK_controller_client_withStateFeedbackZed.py

Result: Not robust, susceptible to wrong pose estimation from ZED --> bad safety action

## Run with value shielding and get pose from Vicon + IMU
In this instance, we will have 3 clients writing data to a single server. So the role of the server and client will be reversed here:
- A vicon client sending vicon pose to server
- A serial client sending serial pose to server
- A control client sending control command to server
- The server reading inputs from all 3 clients and do 1 step forward