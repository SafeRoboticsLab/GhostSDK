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
include CMakeFiles/LimbControl_mb_VISION_0x40.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LimbControl_mb_VISION_0x40.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LimbControl_mb_VISION_0x40.dir/flags.make

CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj: CMakeFiles/LimbControl_mb_VISION_0x40.dir/flags.make
CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj: ../LimbControl/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj -c /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/LimbControl/main.cpp

CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.i"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/LimbControl/main.cpp > CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.i

CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.s"
	/home/buzi/ghost_robotics/arm_toolchain/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/LimbControl/main.cpp -o CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.s

# Object files for target LimbControl_mb_VISION_0x40
LimbControl_mb_VISION_0x40_OBJECTS = \
"CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj"

# External object files for target LimbControl_mb_VISION_0x40
LimbControl_mb_VISION_0x40_EXTERNAL_OBJECTS =

LimbControl_mb_VISION_0x40: CMakeFiles/LimbControl_mb_VISION_0x40.dir/LimbControl/main.cpp.obj
LimbControl_mb_VISION_0x40: CMakeFiles/LimbControl_mb_VISION_0x40.dir/build.make
LimbControl_mb_VISION_0x40: CMakeFiles/LimbControl_mb_VISION_0x40.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LimbControl_mb_VISION_0x40"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LimbControl_mb_VISION_0x40.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold Post-build
	arm-none-eabi-size LimbControl_mb_VISION_0x40
	arm-none-eabi-objcopy -O ihex LimbControl_mb_VISION_0x40 LimbControl_mb_VISION_0x40.bin
	arm-none-eabi-objcopy -O binary -S LimbControl_mb_VISION_0x40 LimbControl_mb_VISION_0x40.bin

# Rule to build all files generated by this target.
CMakeFiles/LimbControl_mb_VISION_0x40.dir/build: LimbControl_mb_VISION_0x40

.PHONY : CMakeFiles/LimbControl_mb_VISION_0x40.dir/build

CMakeFiles/LimbControl_mb_VISION_0x40.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LimbControl_mb_VISION_0x40.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LimbControl_mb_VISION_0x40.dir/clean

CMakeFiles/LimbControl_mb_VISION_0x40.dir/depend:
	cd /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build /home/buzi/Desktop/Princeton/RESEARCH/SAFE/HARDWARE/SPIRIT/GhostSDK_0.12.2/SDK/examples/build/CMakeFiles/LimbControl_mb_VISION_0x40.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LimbControl_mb_VISION_0x40.dir/depend

