# Use Ubuntu as the base image
FROM ubuntu:latest

# Set non-interactive mode during package installation
ENV DEBIAN_FRONTEND=noninteractive
# Set the working directory
WORKDIR /workspace

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libtbb-dev\ 
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Cloning and installing opencv
RUN git clone https://github.com/opencv/opencv.git && \
cd ./opencv && mkdir build && cd build && \
cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
make -j"$(nproc)" && \
make install  && ldconfig

# Cloning and installing intel embree4 to use in th eVcglib
RUN git clone https://github.com/embree/embree.git && \
    cd ./embree && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install  && ldconfig

# Clone VCGlib repository from the devel branch and build FOR NOW I CLONE FROM MY OWN REPO FOR SOME CHANGES IN EmbreeAdaptor.h
# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone -b embree https://github.com/PFasano99/vcglib.git && \
    cd ./vcglib && mkdir build  && cd build 

# Copy the C++ files into the container
COPY ./build_depth.cpp ./build_depth/build_depth.cpp
COPY ./build_depth.h ./build_depth/build_depth.h
COPY ./CMakeLists.txt ./build_depth/CMakeLists.txt

#RUN g++ -o build_depth-cpp ./build_depth/build_depth.cpp -I /usr/local/include/opencv4/ -I /usr/local/include/embree4/ -I ./vcglib/ -I ./vcglib/eigenlib/ -I ./vcglib/wrap/ply/ -L /usr/local/lib/ -lembree4 -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_imgcodecs -fopenmp -O3
RUN cd ./build_depth && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

# Compile and run the C++ files provided as input
CMD ["bash", "-c"]
