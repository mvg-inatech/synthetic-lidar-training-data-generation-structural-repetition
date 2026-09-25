#!/bin/bash
# Run this script after opening the devcontainer to install Python dependencies.

set -e

echo "Installing HELIOS++"

# Note: mamba is installed as a devcontainer feature, so it is not available for execution in the Dockerfile.
micromamba install -y python=3.11
# Python 3.11 is required by bpy (Blender Python API)

micromamba env update -n base -f environment.yml -y
# Note: the only way to install helios++ INCLUDING the Python API is to use conda-forge (in our case via micromamba). The standalone installer doesn't include the Python API.

pip install --no-cache-dir -r requirements.txt

echo ""
echo "Finished"
