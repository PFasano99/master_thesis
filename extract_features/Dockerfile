# Use the official Ubuntu image as the base image
FROM docker.io/pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

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
                && rm -rf /var/lib/apt/lists/*

            
RUN pip install setuptools --upgrade
COPY ./requirements.txt /extract_features/requirements.txt
RUN pip3 install -r /extract_features/requirements.txt
RUN pip install git+https://github.com/gradslam/gradslam.git@conceptfusion

#RUN pip install -e /tesi_image/gradslam/

RUN pip install git+https://github.com/facebookresearch/segment-anything.git 
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install open_clip_torch

#installing DinoV2
RUN pip install --extra-index-url https://pypi.nvidia.com cuml-cu11==23.2.0
RUN pip install git+https://github.com/facebookresearch/dinov2

#this is disabled due to the machine not having a gpu
ENV XFORMERS_DISABLED=1

COPY ./feature_extraction.py /extract_features/feature_extraction.py

CMD ["/bin/bash", "-c"] 
