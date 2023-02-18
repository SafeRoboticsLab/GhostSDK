#!/bin/bash

xhost +

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
  xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
  if [ ! -z "$xauth_list" ]; then
    echo $xauth_list | xauth -f $XAUTH nmerge -
  else
    touch $XAUTH
  fi
  chmod a+r $XAUTH
fi

./docker_clean.bash

no_gui=$1

export REGISTRY_SRC_IMAGE=registry.gitlab.com/ghostrobotics/ghost_gazebo

#docker pull ${REGISTRY_SRC_IMAGE}:release

# updated as of 8/5/20
docker pull ${REGISTRY_SRC_IMAGE}:release_v2

docker run -t -d --name="ghost_simulator" \
  --gpus all \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --env="XAUTHORITY=$XAUTH" \
  ${REGISTRY_SRC_IMAGE}:release_v2

sleep 2

docker exec -it \
  ghost_simulator \
  sh -c "/run_sim.sh $no_gui"
