FROM ubuntu:20.04

RUN apt update && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y software-properties-common build-essential \
    curl wget git \
    python3 python3-distutils python3-dev cmake

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && pip install --upgrade pip \
    && pip install pipx \
    && python3 -m pipx ensurepath

RUN pip install poetry \
    && poetry self add poetry-bumpversion

ENV SHELL /bin/bash

ARG USERNAME=vadmin
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

USER $USERNAME

RUN pip install urllib3 idna certifi
