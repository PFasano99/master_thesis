# Use Paolo.Fasano/tesi_image:cpp as the base image so remember to build the docker image in build_depth folder folder beforhand
FROM Paolo.Fasano/tesi_image:cpp_vertex_dic

COPY ./requirements.txt ./demo_query/requirements.txt
RUN pip3 install -r ./demo_query/requirements.txt
COPY ./text_to_features.py ./demo_query/text_to_features.py    

# Copy the C++ files into the container
COPY ./test_feat.cpp ./demo_query/test_feat.cpp
COPY ./test_feat.h ./demo_query/test_feat.h
COPY ./CMakeLists.txt ./demo_query/CMakeLists.txt

RUN cd ./demo_query && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D    CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j"$(nproc)" && \
    make install

# Compile and run the C++ files provided as input
CMD ["bash", "-c"]
