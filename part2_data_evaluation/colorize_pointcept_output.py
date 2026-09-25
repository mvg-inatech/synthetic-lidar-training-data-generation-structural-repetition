"""
Script to colorize .laz point cloud files based on label values.
Loads files with label_gt and label_pred fields and adds RGB colors based on label mappings.

Example usage:
$ python colorize_pointcept_output.py --in_path pointcept/exp/subway/exp2_sonata_simple_training_pretrained_ft/result/test_preds/2025-10-03_10-21-39/ --out_path CC_TMP
"""

import argparse
from pathlib import Path
import numpy as np
import laspy
from typing import Dict, Tuple
import sys

# Workaround to allow loading pointcept __init__.py files which would otherwise fail due to being in the wrong working directory
sys.path.append("pointcept")
from pointcept.datasets.preprocessing.subway.preprocess_synth_subway import REDUCED_CLASS_ID_TO_NAME


def get_color_map() -> Dict[int, Tuple[int, int, int]]:
    """
    Define the color map for different classes.
    Returns a dict mapping label integers to RGB tuples (0-255 range).
    """
    class_colors = {
        "undefined": [0, 0, 0],
        "beam": [0.03314447030425072, 0.8050270676612854, 0.13163542747497559],
        "cable": [0.0, 0.8088703155517578, 1.0],
        "emergency_exit_notch": [0.2692224681377411, 0.005743207409977913, 1.0],
        "employee_walkway": [0.0009498000727035105, 0.341561496257782, 0.0],
        "ground": [0.4, 0.4, 0.4],
        "lamp": [1.0, 0.3492291271686554, 0.0],
        "pillar": [1.0, 0.6801809072494507, 0.4258980453014374],
        "pole": [1.0, 0.2, 0.2],
        "power_rail": [1.0, 0.8063107132911682, 0.10722056776285172],
        "power_rail_mount": [0.0, 0.005290444940328598, 1.0],
        "rail": [0.9, 0.4, 0.1],
        "rail_tie": [0.0817498192191124, 0.43832260370254517, 0.02392185479402542],
        "rail_tie_area": [0.0817498192191124, 0.43832260370254517, 0.02392185479402542],
        "stairs": [0.07331039756536484, 1.0, 0.08794943243265152],
        "steps": [0.07665728032588959, 0.07665728032588959, 0.07665728032588959],
        "wall": [0.3, 1.0, 0.3],
        "wall_behind_cables": [0.0, 0.1505524218082428, 0.1743464469909668],
    }
    color_map = {class_id: class_colors[class_name] for class_id, class_name in REDUCED_CLASS_ID_TO_NAME.items()}
    return color_map


def colorize_point_cloud(
    input_path: Path,
    output_path: Path,
    use_pred: bool = True,
):
    """
    Load a point cloud, colorize it based on labels, and save.

    Args:
        input_path: Path to input .laz file
        output_path: Path to output .laz file
        use_pred: If True, use label_pred for coloring; otherwise use label_gt
    """
    color_map = get_color_map()

    # Load the point cloud
    las = laspy.read(input_path)

    # Get the labels to use for coloring
    if use_pred:
        labels = las.label_pred
    else:
        labels = las.label_gt

    unique_labels = np.unique(labels)

    # Initialize RGB arrays
    num_points = len(las.points)
    red = np.zeros(num_points, dtype=np.uint16)
    green = np.zeros(num_points, dtype=np.uint16)
    blue = np.zeros(num_points, dtype=np.uint16)

    # Apply colors based on labels
    for label_val in unique_labels:
        mask = labels == label_val
        r, g, b = color_map[label_val]

        # Convert 0..1 to 0..65535 range
        red[mask] = r * 65535
        green[mask] = g * 65535
        blue[mask] = b * 65535

    out_header = laspy.LasHeader(point_format=3)
    out_header.add_extra_dim(laspy.ExtraBytesParams(name="label_gt", type=np.int8))
    out_header.add_extra_dim(laspy.ExtraBytesParams(name="label_pred", type=np.int8))
    out_las = laspy.LasData(out_header)

    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z

    out_las.red = red
    out_las.green = green
    out_las.blue = blue

    out_las.label_gt = las.label_gt
    out_las.label_pred = las.label_pred

    out_las.write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Colorize point cloud files based on label values")
    parser.add_argument("--in_path", type=str, required=True, help="Input directory containing .laz files")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for colorized .laz files")

    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"Error: Input directory does not exist: {in_path}")
        return

    if not in_path.is_dir():
        print(f"Error: Input path is not a directory: {in_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_path}")

    laz_files = list(in_path.glob("*.laz"))
    for i, laz_file in enumerate(laz_files, 1):
        print(f"[{i}/{len(laz_files)}] Processing: {laz_file.name}")

        base_name = laz_file.stem

        # Ground truth colored version
        out_file_gt = out_path / f"{base_name}_gt.laz"
        colorize_point_cloud(laz_file, out_file_gt, use_pred=False)

        # Prediction colored version
        out_file_pred = out_path / f"{base_name}_pred.laz"
        colorize_point_cloud(laz_file, out_file_pred, use_pred=True)

    print("")
    print("Processing completed.")


if __name__ == "__main__":
    main()
