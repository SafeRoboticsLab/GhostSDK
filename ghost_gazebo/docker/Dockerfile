#FROM osrf/ros:melodic-desktop-full
FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# ENVIRONMENT STUFF FOR CUDA
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_ROOT /usr/local/cuda

# recommended from nvidia to use the cuda devices
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
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
    gfortran  && apt autoremove --purge -y && \
               apt clean -y && \
               rm -rf /var/lib/apt/lists/*

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

## install ROS Melodic Desktop Full
# Install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y ros-melodic-desktop && \
    apt-get install -y python-rosinstall &&   apt autoremove --purge -y && \
                                              apt clean -y && \
                                              rm -rf /var/lib/apt/lists/*


## install some of the extra ROS deps
RUN apt-get update && apt-get install -q -y python-catkin-tools \
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
RUN \
  apt-get install -y python3-pip && \
  pip3 install matplotlib==2.0.2 && \
  pip3 install numpy && \
  pip3 install scipy && \
  pip3 install jupyter && \
  pip3 install autograd && \
  pip3 install osqp && \
  pip3 install cython &&   apt autoremove --purge -y && \
                           apt clean -y && \
                           rm -rf /var/lib/apt/lists/*

# Create and install Thirdparties

RUN mkdir -p Thirdparties/usr/local/

# eigen
RUN git clone https://gitlab.com/libeigen/eigen.git && mkdir -p /eigen/build
WORKDIR /eigen/build
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && make -j1 install
WORKDIR /

# osqp
RUN git clone https://github.com/oxfordcontrol/osqp.git
WORKDIR /osqp
RUN git checkout v0.6.1.dev0 && \
    git submodule update --init --recursive && \
    mkdir -p build

WORKDIR /osqp/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
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
                -DPROFILING=OFF && \
                make -j1 && \
                make install
WORKDIR /

# cilantro
RUN git clone https://github.com/kzampog/cilantro.git && mkdir -p /cilantro/build
WORKDIR /cilantro/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j1 && \
    make install
WORKDIR /

# GTSAM
RUN git clone https://github.com/borglab/gtsam.git && mkdir -p /gtsam/build
WORKDIR /gtsam/build
RUN git checkout 4.0.2 && cmake .. \
                          -DCMAKE_BUILD_TYPE=Release \
                          -DGTSAM_INSTALL_CYTHON_TOOLBOX=ON \
                          -DGTSAM_BUILD_TESTS=OFF \
                          -DGTSAM_BUILD_UNSTABLE=OFF \
                          -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
                          -DGTSAM_SUPPORT_NESTED_DISSECTION=OFF \
                          -DGTSAM_WITH_TBB=OFF \
                          -DGTSAM_PYTHON_VERSION=3.6 && \
                          make -j4 && make install && rm -rf /gtsam
WORKDIR /

# OpenCV
RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /opencv_contrib
RUN git checkout 3.4.3
WORKDIR /opencv
RUN git checkout 3.4.3 && mkdir -p build
WORKDIR build
# -DCMAKE_INSTALL_PREFIX=/Thirdparties/usr/local
RUN cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules .. && make -j4 && make install && rm -rf /opencv /opencv_contrib
WORKDIR /

# Create env vars scripts
COPY run_sim.sh /run_sim.sh
RUN chmod +x /run_sim.sh

# remove all the source stuff
RUN rm -rf cilantro osqp opencv opencv_contrib gtsam eigen

# clean the cache
RUN apt-get update && \
  apt autoremove --purge -y && \
  apt clean -y && \
  rm -rf /var/lib/apt/lists/*

EXPOSE 11345

COPY ./ghost_sim_entrypoint.sh /

ENTRYPOINT ["/ghost_sim_entrypoint.sh"]

CMD ["bash"]