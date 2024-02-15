# Use the official Ubuntu image as the base image
FROM docker.io/pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Update the package list
RUN apt-get update && \
    apt-get install -y \
                apt-utils \
                python3 \
                python3-pip \
                python3-dev \
                git \
                build-essential \
                ffmpeg \
                libsm6 \
                libxext6 \
                libblas-dev \
                libatlas-base-dev \
                gcc \
                && rm -rf /var/lib/apt/lists/*


RUN pip install setuptools --upgrade

# Set environment variables to use g++ with C++17
ENV CXX=g++
ENV CXXFLAGS="-std=c++17"

# Copy the requirements.txt file into the container at /app
COPY ./requirements.txt /home/paolo.fasano/tesi_image/requirements.txt
RUN pip3 install -r /home/paolo.fasano/tesi_image/requirements.txt


COPY ./gradslam /home/paolo.fasano/tesi_image/gradslam
COPY ./HoloLens2ForCV /home/paolo.fasano/tesi_image/HoloLens2ForCv
WORKDIR ./home/paolo.fasano/tesi_image

# Install the package from the Git repository
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/facebookresearch/segment-anything.git 
#RUN pip install git+https://github.com/facebookresearch/pytorch3d.git
#RUN pip install git+https://github.com/krrish94/chamferdist.git
RUN pip install chamferdist
RUN pip install open_clip_torch
#RUN pip install git+https://github.com/gradslam/gradslam.git@conceptfusion
RUN pip install /home/paolo.fasano/tesi_image/gradslam/

ENV PATH="/home/paolo.fasano/tesi_image/gradslam/:${PATH}"

COPY ./concept-fusion ./concept-fusion
COPY ./hl2-dump-c60 ./hl2-dump-c60

CMD ["/bin/bash", "-c"] 
# docker run -it --rm --name my-running-script -v "$PWD":/app my-python-app temp.py
# podman image build -t Paolo.Fasano/tesi_image:v0.1 .