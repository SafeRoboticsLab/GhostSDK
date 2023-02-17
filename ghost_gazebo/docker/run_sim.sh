#!/bin/bash

export CATKIN_DIR=/ghost_sim_ws
export ROS_IP=127.0.0.1
export ROS_MASTER_URI=http://127.0.0.1:11311

source /opt/ros/melodic/setup.bash
source ${CATKIN_DIR}/devel/setup.bash

env | grep ROS

use_gui=$1

pkill -f ros
pkill -f rviz
sleep 2

echo "-------------- Starting Roscore ---------------------------"
roscore &
sleep 5

if [ -z "$use_gui" ]
then

echo "-------------- Starting Ghost Autonomy ---------------------------"
roslaunch ghost_master sim_master.launch
else
roslaunch ghost_master sim_master.launch gui:=false
fi