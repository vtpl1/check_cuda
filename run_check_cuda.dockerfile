FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
#FROM ubuntu:18.04
RUN apt update && apt install -y python3-pip curl git
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN pip3 --disable-pip-version-check --no-cache-dir install git+https://github.com/vtpl1/check_cuda.git
#CMD check_cuda
