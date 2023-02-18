#! /bin/bash

echo "Installing generic deps"

sudo apt-get update && \
    apt-get install -y vim \
    apt-utils \
    build-essential \
    psmisc \
    vim-gtk \
    tmux \
    git \
    wget \
    lsb-release \
    lsb-core \
    cmake \
    curl \
    ninja-build \
    libboost-all-dev \
    libtbb-dev \
    gfortran

## install ROS Melodic Desktop Full
# Install ROS
echo "Installing ROS-melodic-desktop-full"
 sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y ros-melodic-desktop && \
    apt-get install -y python-rosinstall


## install some of the extra ROS deps
echo "Installing extra ros dependencies"
sudo apt-get update && sudo apt-get install -q -y python-catkin-tools \
                                            ros-melodic-hector-gazebo-plugins \
                                            ros-melodic-interactive-marker-twist-server \
                                            ros-melodic-joint-state-controller \
                                            ros-melodic-gazebo-plugins \
                                            ros-melodic-effort-controllers \
                                            ros-melodic-controller-manager \
                                            ros-melodic-gazebo-ros-control \
                                            ros-melodic-ros-control \
                                            ros-melodic-joystick-drivers \
                                            ros-melodic-tf2-geometry-msgs \
                                            ros-melodic-pcl-conversions \
                                            ros-melodic-pcl-ros \
                                            ros-melodic-pcl-msgs

# Required python packages for analysis
echo "Installing python packages"
sudo apt-get install -y python3-pip && \
  pip3 install matplotlib==2.0.2 && \
  pip3 install numpy && \
  pip3 install scipy && \
  pip3 install jupyter && \
  pip3 install autograd && \
  pip3 install osqp && \
  pip3 install cython

# Create and install Thirdparties
echo "Installing 3rd parties"

mkdir -p ~/Thirdparties/usr/local/

cd ~/Thirdparties/

# eigen
echo "Installing eigen"
git clone https://gitlab.com/libeigen/eigen.git && mkdir -p eigen/build
cd eigen/build
cmake -DCMAKE_INSTALL_PREFIX=~/Thirdparties/usr/local -DCMAKE_BUILD_TYPE=Release .. && make -j1 install
cd ../..

# osqp
echo "Installing osqp"
git clone https://github.com/oxfordcontrol/osqp.git
cd osqp
git checkout v0.6.1.dev0
git submodule update --init --recursive

mkdir -p build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=~/Thirdparties/usr/local -DCMAKE_BUILD_TYPE=Release \
                -DDFLOAT=OFF \
                -DDLONG=OFF \
                -DQDLDL_FLOAT_TYPE="double" \
                -DQDLDL_INT_TYPE="int" \
                -DQDLDL_BOOL_TYPE="int" \
                -DPRINTING=OFF \
                -DCOVERAGE=OFF \
                -DPYTHON=OFF \
                -DMATLAB=OFF \
                -DR_LANG=OFF \
                -DUNITTESTS=OFF \
                -DPROFILING=OFF
make -j1
make install
cd ../..
ls

# cilantro
echo "Installing cilantro"
git clone https://github.com/kzampog/cilantro.git && mkdir -p cilantro/build
cd cilantro/build
cmake .. -DCMAKE_INSTALL_PREFIX=~/Thirdparties/usr/local -DCMAKE_BUILD_TYPE=Release
make -j1
make install
cd ../..

# GTSAM
echo "Installing GTSAM"
git clone https://github.com/borglab/gtsam.git && mkdir -p gtsam/build
cd gtsam/build
git checkout 4.0.2
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON \
      -DGTSAM_BUILD_TESTS=OFF \
      -DGTSAM_BUILD_UNSTABLE=OFF \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
      -DGTSAM_SUPPORT_NESTED_DISSECTION=OFF \
      -DGTSAM_WITH_TBB=OFF \
      -DGTSAM_PYTHON_VERSION=3.6

make -j4 && make install

cd ../..

# OpenCV
echo "Installing OpenCV"
git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.3
cd ..
cd opencv
git checkout 3.4.3 && mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/Thirdparties/usr/local -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules ..
make -j4
make install
cd ../..
