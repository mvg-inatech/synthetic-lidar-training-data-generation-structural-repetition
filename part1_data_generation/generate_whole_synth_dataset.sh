#!/bin/bash
set -e

python step1_generate_long_tunnel_multiple.py
python step2_export_blender_to_helios.py
python step3_helios_simulate_lidar.py
python step4_generate_deep_learning_input_slices.py

echo ""
echo "Whole synthetic dataset generation done."
