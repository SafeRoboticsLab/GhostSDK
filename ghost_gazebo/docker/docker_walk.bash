#!/bin/bash

docker exec -it \
  ghost_simulator \
  sh -c "source /opt/ros/melodic/setup.bash; rostopic pub --rate=5 /command/setBehaviorId std_msgs/UInt32 \"data: 2\" "
