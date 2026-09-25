FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04


# Install additional OS packages required for building Blender
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install --no-install-recommends git git-lfs build-essential git git-lfs subversion cmake libx11-dev #libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev libwayland-dev wayland-protocols #libxkbcommon-dev libdbus-1-dev linux-libc-dev

# Install convenience tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install --no-install-recommends vim htop btop mc

USER vscode
