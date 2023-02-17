cmake_minimum_required(VERSION 3.5)
project(ghost_gazebo)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    std_msgs
    message_filters
    controller_manager
    joint_state_controller
    robot_state_publisher
    tf
    interactive_markers
    visualization_msgs
)
find_package(gazebo REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(SYSTEM ${Boost_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS} src)
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")


add_executable(link_states_broadcaster src/link_states_broadcaster.cpp)

target_link_libraries(link_states_broadcaster ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

#############
## Install ##
#############

# Install link_states_broadcaster
install(TARGETS
        link_states_broadcaster DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

## Mark executables and/or libraries for installation
install(TARGETS #${PROJECT_NAME} ${PROJECT_NAME}_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# include luanch directory
install(
        DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        FILES_MATCHING PATTERN "*.launch"
)

# include scripts directory
install(
        DIRECTORY worlds
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        FILES_MATCHING PATTERN "*"
)

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
        gazebo_walk.py
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
        README.md
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )