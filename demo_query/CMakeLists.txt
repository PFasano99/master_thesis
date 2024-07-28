cmake_minimum_required(VERSION 3.5)
project(demo_query)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Embree
find_package(embree 4 REQUIRED)
include_directories(${EMBREE_INCLUDE_DIRS})

# Find nlohmann/json
find_package(nlohmann_json REQUIRED)
include_directories(${INTERFACE_INCLUDE_DIRECTORIES})

# Find VCGlib
set(VCG_INCLUDE_DIRS ../vcglib)
set(BUILD_DEPTH_DIR ../build_depth)
set(BUILD_VERTEX_DIR ../build_vertex_dic)
add_subdirectory(${VCG_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/vcglib)

# Include directories
include_directories(
    ${VCG_INCLUDE_DIRS}/
    ${VCG_INCLUDE_DIRS}/eigenlib/
    ${VCG_INCLUDE_DIRS}/wrap/ply/
)

# Set the source file
set(SOURCES
    ./test_feat.cpp
    ./test_feat.h
    ${BUILD_DEPTH_DIR}/build_depth.h
    ${BUILD_VERTEX_DIR}/build_vertex_dic.h
    ${VCG_INCLUDE_DIRS}/wrap/ply/plylib.cpp)

# Add executable target
add_executable(test_feat ${SOURCES})

# Link libraries
target_link_libraries(
    test_feat
    vcglib
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    ${EMBREE_LIBRARY}
    -fopenmp
    -O3
)

# Install executable
install(TARGETS test_feat DESTINATION bin)