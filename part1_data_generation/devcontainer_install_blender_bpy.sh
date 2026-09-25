#!/bin/bash
# Run this script after opening the devcontainer to install the Blender Python API dependency.
# Source: https://developer.blender.org/docs/handbook/building_blender/linux/ (adapted)
BLENDER_VERSION="v4.2.15"

set -e

echo "Installing development environment for Blender $BLENDER_VERSION"


echo ""
echo "Install Initial Packages"
sudo apt update
sudo apt install -y git git-lfs

echo ""
echo "Download Sources"
mkdir ~/blender-git
cd ~/blender-git
git clone https://git.blender.org/blender/blender.git # This cannot be combined with `--branch $BLENDER_VERSION`. This is because ./build_files/linux/install_linux_packages.py will skip downloading SOME precompiled libraries if the Git HEAD is in detached state. It will print this as a notice but complete successfully. The `make bpy` command will fail later due to missing libraries. Therefore, the main branch must be cloned first, followed by a call to the package install script.

echo ""
echo "Install Basic Building Environment"
# Sometimes, the script below hangs infinitely, so we install the packages manually as well.
sudo apt update
sudo apt install -y build-essential git git-lfs subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev
sudo apt install -y libwayland-dev wayland-protocols libxkbcommon-dev libdbus-1-dev linux-libc-dev
# End manual package installation
cd ~/blender-git/blender/
./build_files/linux/install_linux_packages.py

echo ""
echo "Download Libraries"
cd ~/blender-git/blender
make update

echo ""
echo "Update and Build"
cd ~/blender-git/blender
git checkout $BLENDER_VERSION
make update
make bpy

echo ""
echo "Install Blender Python API"
cd ~/blender-git/blender
python build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/
pip install ../build_linux_bpy/bin/bpy-4.2.15-cp311-cp311-manylinux_2_39_x86_64.whl # Path depends on $BLENDER_VERSION
pip install "numpy==1.26.4" # "numpy<2" # because `bpy` does not yet support NumPy v2


echo ""
echo "Finished"