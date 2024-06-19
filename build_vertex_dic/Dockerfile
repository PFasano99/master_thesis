# Use Paolo.Fasano/tesi_image:cpp as the base image so remember to build the docker image in build_depth folder beforhand
FROM Paolo.Fasano/tesi_image:cpp 
# Copy the C++ files into the container
COPY ./build_vertex_dic.cpp ./build_vertex_dic/build_vertex_dic.cpp
COPY ./build_vertex_dic.h ./build_vertex_dic/build_vertex_dic.h
COPY ./CMakeLists.txt ./build_vertex_dic/CMakeLists.txt

RUN cd ./build_vertex_dic && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

# Compile and run the C++ files provided as input
CMD ["bash", "-c"]