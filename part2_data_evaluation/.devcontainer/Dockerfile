FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG USERNAME=vscode

# Match the following two ids with the values from your local dev machine
ARG USER_UID=1000
ARG USER_GID=1000

# Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update and install OS packages
RUN apt-get update && apt-get -y install \
    # Required for devcontainer user setup
    sudo \
    # Basic tools
    wget \
    unzip \
    git \
    curl \
    vim \
    # Convenience tools
    htop \
    mc \
    # YOLO dependencies
    libgl1 \
    libglib2.0-0 && \
    # Clear APT cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add user that will automatically used by vscode
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

# Enable sudo for user
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install UV python package manager, see https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

USER ${USERNAME}
