import argparse
import json
from pathlib import Path
from collections import defaultdict
import laspy
from tqdm import tqdm
from preprocess_synth_subway import REDUCED_CLASS_ID_TO_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze subway dataset splits")
    parser.add_argument("--dataset_root", required=True, help="Path to processed dataset")
    parser.add_argument("--dataset_type", required=True, help="Dataset type (synth or real)")
    return parser.parse_args()


def analyze_split(dataset_root: Path, split_name: str, scene_names: list):
    """Analyze a single split and return class/instance statistics."""
    split_dir = dataset_root / split_name

    class_counts = defaultdict(int)
    class_instances = defaultdict(set)

    for scene_name in tqdm(scene_names, desc=f"Analyzing {split_name}"):
        las_path = split_dir / f"{scene_name}.las"

        las = laspy.read(las_path)
        labels = las.label
        instances = las.instance

        for class_id in REDUCED_CLASS_ID_TO_NAME.keys():
            mask = labels == class_id
            class_counts[class_id] += mask.sum()
            unique_instances = set(instances[mask])
            class_instances[class_id].update(unique_instances)

    return class_counts, class_instances


def print_split_statistics(split_name: str, class_counts: dict, class_instances: dict):
    """Print statistics for a split."""
    total_points = sum(class_counts.values())

    print(f"\n{'=' * 60}")
    print(f"Split: {split_name}")
    print(f"{'=' * 60}")
    print(f"Total points: {total_points:,}")
    print()
    print(f"{'Class':<30} {'Points':>15} {'%':>8} {'Instances':>12}")
    print("-" * 67)

    for class_id in sorted(REDUCED_CLASS_ID_TO_NAME.keys()):
        class_name = REDUCED_CLASS_ID_TO_NAME[class_id]
        count = class_counts[class_id]
        percentage = (count / total_points * 100) if total_points > 0 else 0
        num_instances = len(class_instances[class_id])
        print(f"{class_id:2d} {class_name:<27} {count:>15,} {percentage:>7.2f}% {num_instances:>12,}")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_type = args.dataset_type

    info_path = dataset_root / f"{dataset_type}_dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {info_path}")

    with open(info_path, "r") as f:
        dataset_info = json.load(f)

    splits = dataset_info["splits"]

    if dataset_type == "synth":
        split_names = [f"{dataset_type}_train", f"{dataset_type}_val", f"{dataset_type}_test"]
    elif dataset_type == "real":
        split_names = [f"{dataset_type}_train", f"{dataset_type}_test"]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    for split_name in split_names:
        scene_names = splits[split_name]
        class_counts, class_instances = analyze_split(dataset_root, split_name, scene_names)
        print_split_statistics(split_name, class_counts, class_instances)


if __name__ == "__main__":
    main()
