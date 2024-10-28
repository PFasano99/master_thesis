# Use Paolo.Fasano/tesi_image:cpp as the base image so remember to build the docker image in build_depth folder beforhand
FROM Paolo.Fasano/tesi_image:cpp 

# Cloning and installing nlohmann/json to use in the build_vertex_dic.h to read json
RUN git clone https://github.com/nlohmann/json.git && \
    cd ./json && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install  && ldconfig

# Download and install LibTorch
RUN apt-get update && apt-get install -y wget \ 
        unzip \
        python3-dev \
        python3-pip 
        
# Install Python packages
# Copy the requirements.txt file into the container at /app
COPY ./requirements.txt ./build_vertex_dic/requirements.txt
RUN pip3 install -r ./build_vertex_dic/requirements.txt

# Copy the C++ files into the container
COPY ./build_vertex_dic.cpp ./build_vertex_dic/build_vertex_dic.cpp
COPY ./build_vertex_dic.h ./build_vertex_dic/build_vertex_dic.h
COPY ./build_vertex_unit_tests.h ./build_vertex_dic/build_vertex_unit_tests.h
COPY ./CMakeLists.txt ./build_vertex_dic/CMakeLists.txt

COPY ./bin_to_png.py ./build_vertex_dic/bin_to_png.py
COPY ./hl2_test.py ./build_vertex_dic/hl2_test.py

RUN cd ./build_vertex_dic && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

# Compile and run the C++ files provided as input
CMD ["bash", "-c"]
