cmake_minimum_required(VERSION 3.5)
project(build_vertex_dic)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Embree
find_package(embree 4 REQUIRED)
include_directories(${EMBREE_INCLUDE_DIRS})

# Find VCGlib
#find_package(vcglib REQUIRED)
#include_directories(${VCG_INCLUDE_DIRS})
set(VCG_INCLUDE_DIRS ../vcglib)
set(BUILD_DEPTH_DIR ../build_depth)
add_subdirectory(${VCG_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/vcglib)


# Include directories
include_directories(
    ${VCG_INCLUDE_DIRS}/
    ${VCG_INCLUDE_DIRS}/eigenlib/
    ${VCG_INCLUDE_DIRS}/wrap/ply/
)

# Set the source file
set(SOURCES
    ./build_vertex_dic.cpp
    ./build_vertex_dic.h
    ${BUILD_DEPTH_DIR}/build_depth.h
    ${VCG_INCLUDE_DIRS}/wrap/ply/plylib.cpp)

# Add executable target
add_executable(build_vertex_dic ${SOURCES})

# Link libraries
target_link_libraries(
    build_vertex_dic
    #build_depth
    vcglib
    ${OpenCV_LIBS}
    ${EMBREE_LIBRARY}
    -fopenmp
    -O3
)

# Install executable
install(TARGETS build_vertex_dic DESTINATION bin)