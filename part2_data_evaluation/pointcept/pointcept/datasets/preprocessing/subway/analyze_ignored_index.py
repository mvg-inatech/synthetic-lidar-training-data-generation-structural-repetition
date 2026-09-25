import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


def analyze_file(file_path):
    """
    Load a .pth file and calculate the percentage of points with segment == -1
    Returns: (file_path, percentage, total_points)
    """
    try:
        data = torch.load(file_path, map_location="cpu")
        segment = data["segment"]

        total_points = len(segment)
        undefined_points = np.sum(segment == -1)
        percentage = (undefined_points / total_points) * 100 if total_points > 0 else 0.0

        return str(file_path), percentage, total_points, undefined_points

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return str(file_path), 0.0, 0, 0


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze undefined points in preprocessed point cloud dataset")
    parser.add_argument("--dataset_root", required=True, help="Path to preprocessed dataset root (containing train/val/test folders)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {dataset_root}")

    # Find all .pth files in all subdirectories
    pth_files = list(dataset_root.rglob("*.pth"))

    if not pth_files:
        raise ValueError(f"No .pth files found in {dataset_root}")

    print(f"Found {len(pth_files)} .pth files")
    print("=" * 80)

    results = []
    total_points_overall = 0
    total_undefined_overall = 0

    # Process each file
    for pth_file in tqdm(pth_files, desc="Analyzing files"):
        file_path, percentage, total_points, undefined_points = analyze_file(pth_file)
        results.append((file_path, percentage, total_points, undefined_points))

        total_points_overall += total_points
        total_undefined_overall += undefined_points

        # print(f"{file_path}: {percentage:.2f}% ({undefined_points}/{total_points} points)")

    print("=" * 80)

    # Overall statistics
    overall_percentage = (total_undefined_overall / total_points_overall) * 100 if total_points_overall > 0 else 0.0
    print("Overall statistics:")
    print(f"Total files: {len(results)}")
    print(f"Total points: {total_points_overall:,}")
    print(f"Total undefined points: {total_undefined_overall:,}")
    print(f"Overall percentage of undefined points: {overall_percentage:.2f}%")
    print()

    # Sort results by percentage
    results.sort(key=lambda x: x[1], reverse=True)

    # Top 10 highest percentages
    print("=" * 80)
    print("TOP 10 HIGHEST PERCENTAGES OF UNDEFINED POINTS:")
    print("=" * 80)
    for i, (file_path, percentage, total_points, undefined_points) in enumerate(results[:10]):
        filename = Path(file_path).name
        print(f"{i + 1:2d}. {filename:<30} {percentage:6.2f}% ({undefined_points:,}/{total_points:,})")

    print()

    # Top 10 lowest percentages
    print("=" * 80)
    print("TOP 10 LOWEST PERCENTAGES OF UNDEFINED POINTS:")
    print("=" * 80)
    results_lowest = sorted(results, key=lambda x: x[1])
    for i, (file_path, percentage, total_points, undefined_points) in enumerate(results_lowest[:10]):
        filename = Path(file_path).name
        print(f"{i + 1:2d}. {filename:<30} {percentage:6.2f}% ({undefined_points:,}/{total_points:,})")

    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
