cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0048 NEW)
project(wildcat_slam VERSION 0.0.1)



#################### Set compiler options begin ####################
set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wthread-safety")
endif()
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
option(ENABLE_AGGRESIVE_OPT "Enable aggressive optimization" ON)
if(ENABLE_AGGRESIVE_OPT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
#################### Set compiler options end ####################



#################### Find dependencies begin ####################
find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    rosbag
    std_msgs
    tf)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(Protobuf REQUIRED)
if (Protobuf_VERBOSE)
    message(STATUS "Using Protocol Buffers ${protobuf_VERSION}")
endif ()
find_package(fmt REQUIRED)

add_subdirectory(3rd-party/abseil-cpp)
#################### Find dependencies end ####################



#################### generate proto srcs and headers ####################
file(GLOB_RECURSE ALL_PROTOS "proto/*.proto")

# TODO(cschuet): Move proto compilation to separate function.
set(ALL_PROTO_SRCS)
set(ALL_PROTO_HDRS)
foreach(ABS_FIL ${ALL_PROTOS})
    file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
    get_filename_component(DIR ${REL_FIL} DIRECTORY)
    get_filename_component(FIL_WE ${REL_FIL} NAME_WE)

    list(APPEND ALL_PROTO_SRCS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc")
    list(APPEND ALL_PROTO_HDRS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc"
            "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h"
        COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --cpp_out  ${PROJECT_BINARY_DIR} -I
        ${PROJECT_SOURCE_DIR} ${ABS_FIL}
        DEPENDS ${ABS_FIL}
        COMMENT "Running C++ protocol buffer compiler on ${ABS_FIL}"
        VERBATIM
    )
endforeach()
set_source_files_properties(${ALL_PROTO_SRCS} ${ALL_PROTO_HDRS} PROPERTIES GENERATED TRUE)
#################### generate proto srcs and headers ####################



catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
    DEPENDS EIGEN3 PCL
)

include_directories(
    src
    3rd-party/Sophus-1.22.10
    gen/cpp
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

set(
    PROJECT_SRCS
    src/common/utils.cc
    src/common/msg_conversion.cc
    src/common/time.cc
    src/common/histogram.cc
    src/odometry/lidar_odometry.cc
    src/odometry/surfel_extraction.cc
    src/odometry/knn_surfel_matcher.cc
    src/odometry/cost_functor.cc
    src/odometry/surfel.cc
)
list(APPEND PROJECT_SRCS ${ALL_PROTO_SRCS})

add_executable(wildcat_slam_node
    src/wildcat_slam_node.cc
    ${PROJECT_SRCS}
)
set(
    TEST_EXECUTABLE_COMMON_DEPS
    absl::strings
    absl::strings_internal
    absl::str_format
    absl::variant
    absl::synchronization
    absl::flat_hash_set
    fmt
    glog
    ${catkin_LIBRARIES}
    ${CERES_LIBRARIES}
)

target_link_libraries(wildcat_slam_node ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${Protobuf_LIBRARIES} ${TEST_EXECUTABLE_COMMON_DEPS})

# message("testkk " ${PROJECT_SRCS})
include(cmake/google-test.cmake)
set(TEST_LIB wildcat_core)
add_test_library_srcs("${PROJECT_SRCS}")
enable_automatic_test_and_benchmark()
