# Ghost Gazebo

Ghost Robotics supporting software for simulating the Vision60 platform in Gazebo.

For more docs about the Ghost Robotics Software system, please look [here](https://ghostrobotics.gitlab.io/docs/gazebo.html)

This repository depends explicitly on the [ghost_description](https://gitlab.com/ghostrobotics/ghost_description) repo.

If building from source, be sure to clone ghost_description in the same catkin workspace and
build it.

##  Running The Stack

### Docker Support (Recommended for HL in Sim!!)

We now provide a self-sufficient docker container that, in conjunction with nvidia-docker
, contains all of the dependencies for the Ghost High-Level Stack. This container is part of the
gitlab container registry associated with this repo. If you'd like to run this, you will still
need to contact Ghost Robotics to get access to the ghost_gazebo container registry.

Dependencies: Install docker [here](https://docs.docker.com/engine/install/ubuntu/) and nvidia-docker [here](https://github.com/NVIDIA/nvidia-docker).

In order to pull and run the container, first make sure to `docker login registry.gitlab.com/ghostrobotics/ghost_gazebo`.

This will prompt you to use your credentials; because the Ghost Robotics Organization requires
2FA, you will have to create a personal access token on the gitlab website (under the same
user that has access to the `ghost_gazebo` repo). Instructions [here](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html)

Once you have that token, when prompted to login above, use the token as your password and your
usual username as the username.

Now clone this repository and cd into `ghost_gazebo/docker`. Several scripts are provided to get
 up and running with the ghost_gazebo docker image:

```
Dockerfile          <-- Dockerfile for all of the dependencies; building this will NOT give you the Ghost high-level and low-level releases.
run_ghost_sim.bash  <-- the main script that will pull the container and run it with nvidia-docker
run_sim.sh          <-- this script is called inside the container, here for convenience.

# every script below here is for interacting with the sim in the container
# these are ran on you local machine
docker_sit.bash     <-- Make the robot sit
docker_stand.bash   <-- Make the robot stand
docker_walk.bash    <-- Make the robot walk
docker_enable_waypoint.bash   <-- put the robot in waypoint mode (can now send from rviz)
docker_clean.bash   <-- run in order to stop and rm the container
```

### Building from Source
The system has been tested exclusively on Ubuntu 18.04 with ROS Melodic and Gazebo9 & Gazebo11.
If you haven't already, install those now.

ROS Melodic ships natively with Gazebo9, but supports gazebo11, which can yield ~10% performance
 improvements, if desired. Some helpful links on that
 [here](http://gazebosim.org/tutorials?tut=install_ubuntu) and
 [here](http://gazebosim.org/tutorials/?tut=ros_wrapper_versions).

 ROS Dependencies
```shell script
sudo apt install -y ros-melodic-interactive-marker-twist-server ros-melodic-joint-state-controller ros-melodic-gazebo-plugins ros-melodic-effort-controllers ros-melodic-controller-manager ros-melodic-gazebo-ros-control ros-melodic-ros-control ros-melodic-joystick-drivers ros-melodic-hector-gazebo-plugins
```

Optional: [Tmux](https://linuxize.com/post/getting-started-with-tmux/) is a handy terminal
multiplexer that is quite useful when working with ROS.

### Run
 1. Acquire the Ghost Robotics Gazebo SDK artifacts, including the binaries `vision3_gazebo` and
  `spirit_gazebo` as well as the `libgazebo.a` artifact. The next set of instructions/information
   is around using the default binary as a standalone ROS node that controls the robot.
 2. Copy the `ghost_gazebo` repo from the root of the artifacts directory to a catkin workspace (or unzip the artifacts in one)
 3. Copy the two binary executables from the `bin` directory of the SDK release into the
    root directory of `ghost_gazebo`. This allows us to launch the SDK from roslaunch.
 3. `catkin build ghost_description ghost_gazebo`
 4. source the `devel/setup.bash`
 5. `roslaunch ghost_gazebo vision60_gazebo.launch | spirit40_gazebo.launch` assuming the
  executables from the GhostSDK were copied successfully, these launch files will launch the low
  -level controller as well.

That's it! You should now see a gazebo window with a stylish legged robot, and
the executable wrapping the SDK in ROS should also be running.

A good sanity test would be sending a stand command from the command line:
`rostopic pub --rate=5 /behaviorMode std_msgs/UInt32 "data: 1"`

By default the instructions above will run the simulator in standalone mode, meaning that none of
the Ghost High-Level Autonomy is running, only the Low-Level controller. For instructions on
running the Gazebo simulator with the Autonomy release, please see the `gazebo.md` document in
that archive.

### Rosparams
A note on the rosparams available for the SDK executable, set in the launch files:

```xml
<rosparam subst_value="True">
    publish_state: $(arg publish_state)  <!-- 1 = standalone mode, 0 = with Ghost Autonomy -->
    gazebo_topic_prefix: $(arg gazebo_topic_prefix) <!-- Defaults to /vision60 or /spirit -->
</rosparam>

```

`publish_state`  controls whether the Low-Level will run standalone, or with the Ghost High-Level
Autonomy release. `1` = Standalone, `0` = autonomy; note that running the simulator with the
autonomy system requires the Ghost High-Level Release.

`model name` is the name you want to spawn the robot under in gazebo. This will make also set the top level namespace
for the nodes and topics used in controlling this robot, everything will be prefixed by `/model_name/`. This applies to
the sensor output from the gazebo as well as the TF tree for this robot. The default is `vision60`.

## URDF/SDF Conversions

All of these files are in the `ghost_description` repo.

Note that if you'd like to generate a urdf for the V4, please set the v4 property in the
`vision60_gazebo.urdf.xacro` to 1. Default V3 and V3 URDFs (not urdf.xacro) are provided in the
`./urdf/generated` directory

 ```xml
 <!--  This defaults to 0, set it to 1 -->
 <xacro:property name="v4" value="0"/> <!-- 0=v3, 1=v4, selects meshes if enabled -->
```

The controllers are now spawned from the `xacro` files, but in case there is still interest in
using SDF's, conversion instructions are provided.

SDFs are already built, but in case you'd like to make changes to the original urdf.xacro, here's
the quick summary for re-making the SDFs.

```
 cd urdf
 xacro vision60_gazebo.urdf.xacro > vision60_gazebo.urdf
 gz sdf -p vision60_gazebo.urdf > ../sdf/vision60_gazebo.sdf
```

If just `xacro` doesn't run, try `rosrun xacro xacro`.

### A Word on Performance
Using an, i7-9750H and an RTX 2080 Max-Q on a laptop, we're seeing the sim run at about 65% of
real time, when using gazebo9. Using Gazebo11, this rises to about 75% with nominally the same
settings. If you'd like to tweak performance for your machine/application, the main parameters
to adjust are located in each of the world files..

```$xslt
   <physics type="ode">
        <!--  0.0003, and 200 works (0.85 rtf) but not amazing, still has some osc. -->
        <!--  0.0003, and 300 is much better (0.65 rtf) -->

        <max_step_size>0.0003</max_step_size>
        <real_time_update_rate>-1</real_time_update_rate>
        <ode>
            <solver>
                <iters>300</iters>
            </solver>
        </ode>
    </physics>
```


## ROS Structure

If using the sim in combination with the Ghost Robotics High Level Release, the simulation node
is treated exactly the same way as the robot. Users do not need to interact with the sim node
 when this is the case, they can interact with the high level as they would with a normal robot.

If using the sim node as a standalone method for controlling the robot, the topic structure is
 given below. Take a look/run the `gazebo_walk.py` script in this repo. It
 provides sample publishers for all of the actions.

| topic name | type | publisher | subscriber | Notes |
|--------|--------|--------|--------|--------|
|/pose | Pose | User | /GhostGazeboNode | In look around mode, control attitude |
|/twist | Twist | User | /GhostGazeboNode | In walk mode...walk! |
|/behaviorId | UInt32 | User | /GhostGazeboNode | Change BehaviorId (look at the python example for
 more
 info) |
|/behaviorMode | UInt32 | User | /GhostGazeboNode | Change BehaviorMode |

## Potential Improvements
*  Multi-threaded physics
*  Perception Sensors <-- this is in progrss, there is one, wide FOV depth camera at the front
*  Autonomous Examples <-- this runs with the high level release
*  Footstep planning support <-- also runs with the high level release