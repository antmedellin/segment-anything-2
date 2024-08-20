# build command  
# docker build -t sam2 .  

FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install -y \
    git \
    build-essential \
    wget \
    unzip \
    pkg-config \
    cmake \
    pip \
    sudo \
    g++ \
    g++-9 \
    ca-certificates \
    libgl1-mesa-glx \
    gdal-bin \
    htop \
    nano \
    curl \
    software-properties-common

RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get install -y python3.11-tk

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11


RUN pip install \
    matplotlib \
    ipykernel \
    opencv-python \
    scikit-learn \
    albumentations \
    transformers \
    evaluate \
    numpy \
    ninja

RUN git clone https://github.com/facebookresearch/segment-anything-2 && \
    cd segment-anything-2 && \
    python3 -m pip install -e . && \
    python3 -m pip install -e ".[demo]"  
    # python3 setup.py build_ext --inplace

# RUN pip install \
#     torch \
#     torchvision\
#     omegaconf \
#     torchmetrics==0.10.3 \
#     fvcore \
#     iopath \
#     xformers==0.0.18 \
#     submitit 
#     # cuml-cu11
    

# RUN pip install \
#     lightning \
#     tensorboard \
#     torch-tb-profiler \
#     pandas \
#     matplotlib \
#     seaborn 

# RUN pip install --upgrade torchmetrics 

# RUN apt-get install -y python3-tk

# RUN pip install segmentation-models-pytorch 






RUN useradd -m sam2_user 

RUN echo "sam2_user ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sam2_user

USER sam2_user

WORKDIR /home