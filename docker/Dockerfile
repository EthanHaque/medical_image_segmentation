FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg


RUN conda update conda && \
        conda create -y -n ffcv-ssl python=3.9 \
        pytorch \
        torchvision \
        pytorch-cuda=11.8 \
         -c pytorch -c nvidia

RUN conda install -y -n ffcv-ssl \
        cupy \
        pkg-config \
        compilers \
        libjpeg-turbo \
        opencv \
        numba \
        torchmetrics \
        numpy \
        -c conda-forge

RUN conda run -n ffcv-ssl pip install matplotlib \
    pytorch-lightning \
    pydicom \
    rich \
    pillow \
    python-gdcm \
    pylibjpeg[all] \
    nibabel \
    segmentation-models-pytorch

RUN git clone https://github.com/EthanHaque/FFCV-SSL.git
RUN conda run -n ffcv-ssl pip install -e FFCV-SSL

