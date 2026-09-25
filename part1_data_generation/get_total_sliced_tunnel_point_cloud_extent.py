import argparse
import glob
import os

import laspy
import numpy as np


def compute_slice_length(laz_file: str) -> float:
    """
    Compute the length of a tunnel slice from its point cloud.

    Uses PCA to find the principal axis of the point cloud, then returns
    the extent of the points projected onto that axis.

    Args:
        laz_file: Path to the .laz file

    Returns:
        Length of the slice in meters
    """
    las = laspy.read(laz_file)
    points = np.vstack((las.x, las.y, las.z)).T

    if len(points) < 2:
        return 0.0

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis is the eigenvector with the largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    projections = centered @ principal_axis
    return float(projections.max() - projections.min())


def main():
    parser = argparse.ArgumentParser(description="Compute total length of sliced tunnel point clouds")
    parser.add_argument(
        "--in_path",
        default="sliced_tunnels",
        help="Path to the directory containing the sliced tunnel .laz files",
    )
    args = parser.parse_args()

    laz_pattern = os.path.join(args.in_path, "*.laz")
    laz_files = sorted(glob.glob(laz_pattern))

    if not laz_files:
        print(f"No .laz files found in {args.in_path}")
        return

    print(f"Found {len(laz_files)} slices in {args.in_path}\n")

    lengths = []
    for laz_file in laz_files:
        length = compute_slice_length(laz_file)
        filename = os.path.basename(laz_file)
        print(f"  {filename:<60} length: {length:.2f} m")
        lengths.append(length)

    lengths = np.array(lengths)
    print(f"\nTotal length across {len(laz_files)} slices: {lengths.sum():.2f} m")
    print(f"Mean:   {lengths.mean():.2f} m")
    print(f"Stddev: {lengths.std():.2f} m")
    print(f"Min:    {lengths.min():.2f} m")
    print(f"Max:    {lengths.max():.2f} m")


if __name__ == "__main__":
    main()
