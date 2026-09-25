"""
Standalone script to compute the precise context spline length of the tunnel path curve in every .blend file.
"""

import argparse
import json
from pathlib import Path

import bpy
from tqdm import tqdm

PATH_LENGTH_RESOLUTION = 1000


def parse_args():
    # Blender passes its own args before "--"; only parse what comes after it.
    parser = argparse.ArgumentParser(description="Compute total Bezier path length across .blend files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to scan for .blend files (non-recursive).")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save the output JSON. Default: 'tunnel_lengths.json' inside input_dir.")
    return parser.parse_args()


def find_blend_files(input_dir: Path):
    """Return .blend files directly inside input_dir, sorted ascending. No subdirectory recursion."""
    blend_files = sorted(input_dir.glob("*.blend"))
    return blend_files


def get_context_spline_path_object():
    """Return the single curve object to use as the tunnel path."""
    curve_objects = [obj for obj in bpy.data.objects if obj.type == "CURVE"]
    assert len(curve_objects) == 1, f"Expected exactly one curve object, found {len(curve_objects)}: {[o.name for o in curve_objects]}"
    return curve_objects[0]


def compute_path_length(path_obj):
    """Compute the arc length of the path object's first spline."""
    spline = path_obj.data.splines[0]
    return spline.calc_length(resolution=PATH_LENGTH_RESOLUTION)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json) if args.output_json else input_dir / "tunnel_lengths.json"

    blend_files = find_blend_files(input_dir)
    assert blend_files, f"No .blend files found in {input_dir}"

    seed_lengths = {}
    for blend_file in tqdm(blend_files, desc="Computing path lengths"):
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
        path_obj = get_context_spline_path_object()
        context_spline_length = compute_path_length(path_obj)
        seed_lengths[blend_file.name] = context_spline_length
        print(f"Computed context spline length for {blend_file.name}: {context_spline_length} m")

    total_length = sum(seed_lengths.values())
    print(f"Total accumulated context spline length: {total_length} m")

    with open(output_json, "w") as f:
        json.dump({"seed_lengths": seed_lengths, "total_length": total_length}, f, indent=4)
    print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
