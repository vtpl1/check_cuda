FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# FROM nvcr.io/nvidia/deepstream:5.0-20.07-devel
#FROM ubuntu:18.04
RUN apt update && apt install -y python3-pip python3-dev curl git ninja-build build-essential gdb
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-4 && apt update && apt install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

COPY requirements.dev.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.dev.txt \
    && rm -rf /tmp/pip-tmp

COPY requirements.prod.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.prod.txt \
    && rm -rf /tmp/pip-tmp


RUN mkdir /tmp/decord_build && cd /tmp/decord_build && git clone --recursive https://github.com/dmlc/decord

ADD ./Video_Codec_SDK_11.0.10.tar.xz /tmp/decord_build/
RUN cp -r /tmp/decord_build/Video_Codec_SDK_11.0.10/Lib/linux/stubs/x86_64/*.so /usr/local/cuda/lib64/ && rm -rf /tmp/decord_build/Video_Codec_SDK_11.0.10

RUN cd /tmp/decord_build/ && cd decord && mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && make && make install && cd ../python && python3 setup.py install && rm -rf /tmp/decord_build

COPY requirements.experiment.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.experiment.txt \
    && rm -rf /tmp/pip-tmp

ARG USERNAME=vadmin
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME
