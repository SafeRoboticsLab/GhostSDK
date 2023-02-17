import pyzed.sl as sl

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

while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(zed_pose.get_translation(py_translation).get()[0], 3) - 0.277
        ty = round(zed_pose.get_translation(py_translation).get()[1], 3) - 0.06
        tz = round(zed_pose.get_translation(py_translation).get()[2], 3) - 0.15
        print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

