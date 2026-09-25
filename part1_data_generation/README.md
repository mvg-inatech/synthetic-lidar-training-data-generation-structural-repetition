# Synthetic training data generation for Scan-to-BIM

* Uses the standalone Blender Python API and the Helios++ LiDAR simulator
* Generates long synthetic subway tunnels based on short tunnel segments

## Installing dependencies

1. Install Visual Studio Code
1. Install the [devcontainer extension](https://code.visualstudio.com/docs/devcontainers/containers)
1. Open this folder in the devcontainer.
1. `$ ./devcontainer_install_python.sh`
1. `$ ./devcontainer_install_blender_bpy.sh` (Will take approximately 30 minutes to complete, due to the official Blender build process being slow.)

## Updating HELIOS++ (conda/mamba) dependencies (development purposes only)
1. Rebuild the devcontainer
1. Instead of running `devcontainer_install_python.sh`, install the Python dependencies manually:
    1. `$ micromamba install -y python=3.11`
    1. `$ micromamba install -c conda-forge helios gdal sqlite`
1. Freeze the current Python environment and save it for future use: `$ micromamba env export > environment.yml`. In the updated `environment.yml` file, remove the added `pip:` section, as pip packages are managed separately via `requirements.txt` and should not be installed through mamba.


## Generating the whole synthetic Scan-to-BIM dataset for subway tunnels

You can either manually run the steps described in the sections below, or use this convenience script that runs all steps automatically:
1. `$ ./generate_whole_synth_dataset.sh`


## Step 1: Acquire Blender scene (generate the synthetic long tunnel mesh dataset)

**Running**

1. `$ python step1_generate_long_tunnel_multiple.py`


### Variant: using existing Blender files from the video game

Instead of generating synthetic long tunnels from single tunnel segment files, you can load an existing scene into the data processing pipeline.
First, acquire the `.blend` file.
Then, you must split the Blender scene into loose parts.

#### Split the Blender scene into loose parts using the Blender Python API (bpy)

The tunnel segment library consists of `.blend` files that contain parts, such as rails.
However, when selecting, e.g., a rail part in Blender, it often contains other rail parts at different poses in space.
For accurate instance segmentation annotations, we need to split these objects that contain multiple parts into individual objects.

This can be done in Blender itself using the GUI:
1. Select all (press A)
1. Enter edit mode (press TAB)
1. Open split menu (press P)
1. Split by loose parts

However, for large scenes, this operation causes Blender to freeze. Therefore, this script provides a progress bar and a time estimate.


**Running**
1. `$ python blender_split_loose_parts.py`



## Step 2: Export the Blender scene into one .obj file per object

In order for HELIOS++ to output instance annotations in the point cloud, each instance must be loaded from a separate `.obj` file.
Therefore, this step exports the Blender scene into one `.obj` file per object.
It resembles the functionality of this [Blender plugin](https://github.com/neumicha/Blender2Helios) as a standalone application, since the Blender UI becomes slow for large scenes.

### Running
1. `$ python step2_export_blender_to_helios.py`


## Step 3: Simulate the LiDAR scanner with HELIOS++

Runs the [HELIOS++ LiDAR Simulator](https://www.geog.uni-heidelberg.de/gis/helios.html) by the University of Heidelberg, using Python.


### Running
1. `$ python step3_helios_simulate_lidar.py`

## Step 4: Slice the generated tunnel point clouds along the tunnel axis

To avoid out-of-memory errors and to make the spatial extent of the synthetic point cloud input comparable to the real-world point cloud input captured by the mobile robot, the generated long tunnel point clouds need to be sliced along the tunnel axis.
The slice length is sampled from a Gaussian distribution with a fixed mean, and is measured along the simulated robot trajectory (rather than along the tunnel axis).

### Running
1. `$ python step4_generate_deep_learning_input_slices.py`
