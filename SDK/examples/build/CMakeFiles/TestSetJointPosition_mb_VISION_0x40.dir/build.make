# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/flags.make

CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj: CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/flags.make
CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj: ../TestSetJointPosition/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj -c /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/TestSetJointPosition/main.cpp

CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.i"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/TestSetJointPosition/main.cpp > CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.i

CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.s"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/TestSetJointPosition/main.cpp -o CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.s

# Object files for target TestSetJointPosition_mb_VISION_0x40
TestSetJointPosition_mb_VISION_0x40_OBJECTS = \
"CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj"

# External object files for target TestSetJointPosition_mb_VISION_0x40
TestSetJointPosition_mb_VISION_0x40_EXTERNAL_OBJECTS =

TestSetJointPosition_mb_VISION_0x40: CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/TestSetJointPosition/main.cpp.obj
TestSetJointPosition_mb_VISION_0x40: CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/build.make
TestSetJointPosition_mb_VISION_0x40: CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TestSetJointPosition_mb_VISION_0x40"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold Post-build
	arm-none-eabi-size TestSetJointPosition_mb_VISION_0x40
	arm-none-eabi-objcopy -O ihex TestSetJointPosition_mb_VISION_0x40 TestSetJointPosition_mb_VISION_0x40.bin
	arm-none-eabi-objcopy -O binary -S TestSetJointPosition_mb_VISION_0x40 TestSetJointPosition_mb_VISION_0x40.bin

# Rule to build all files generated by this target.
CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/build: TestSetJointPosition_mb_VISION_0x40

.PHONY : CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/build

CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/clean

CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/depend:
	cd /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/TestSetJointPosition_mb_VISION_0x40.dir/depend

