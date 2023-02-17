
cmake_minimum_required(VERSION 3.5.0)

set(MAVLINK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/mavlink" CACHE STRING "mavlink location")
set(PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/pybind11" CACHE STRING "pybind11 location")
set(PYBIND ON CACHE BOOL "Set ON to build python binding")

project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)

else()
  add_compile_options(-Wno-address-of-packed-member -fvisibility=hidden)
endif()

# Produces a static library by this name
set(libname grmblink)
set(execname main)
set(pymodulename grmblinkpy)

set(SRC_LIB
	${CMAKE_CURRENT_SOURCE_DIR}/src/mblink.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/mblink_protocol.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/mblink_socket.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/app_interface.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/data_dicts.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/modemgr/modemgr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/modemgr/modemgr_client.cpp
	)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modemgr)

# include mavlink
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty)

# include eigen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/eigen)

add_library(${libname} STATIC ${SRC_LIB})
target_compile_definitions(${libname} INTERFACE -DEIGEN_NO_DEBUG)

if (WIN32)
  target_link_libraries(${libname} ws2_32)
else()
  target_link_libraries(${libname} pthread)
endif()

if (${PYBIND})
  # pybind11
  add_subdirectory(${PYBIND11_DIR} "pybind11")
  set(SRC_PYBIND
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mblinkpy_bindings.cpp" 
    )
  pybind11_add_module(${pymodulename} ${SRC_PYBIND})
  target_link_libraries(${pymodulename} PRIVATE ${libname})
  # Need PIC for linking dynamic python binding library
  set_target_properties(${libname} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

