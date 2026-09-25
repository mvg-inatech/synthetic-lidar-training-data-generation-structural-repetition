import os
import argparse
import sys
import traceback
import numpy as np
from pathlib import Path
import laspy
from tqdm import tqdm
import random
import json
import open3d as o3d
from multiprocessing import cpu_count
import multiprocessing as mp
import signal
from scipy.spatial import cKDTree

REDUCED_CLASS_ID_TO_NAME = {
    0: "undefined",
    1: "ground",
    2: "rail",
    3: "wall",
    4: "power_rail",
    5: "power_rail_mount",
    6: "pole",
}

# key: fully annotated synthetic dataset class id
# value: reduced annotation dataset class id
ORIGINAL_TO_REDUCED_CLASS_ID_MAPPING = {
    -1: 0,  # clutter/undefined
    0: 1,  # "ground"
    1: 3,  # "wall"
    2: 2,  # "rail"
    3: 1,  # "rail_tie_area" -> "ground"
    4: 5,  # "power_rail_mount"
    5: 4,  # "power_rail"
    6: 0,  # "cable" -> "undefined"
    7: 3,  # "wall_behind_cables" -> "wall"
    8: 3,  # "emergency_exit_notch" -> "wall"
    9: 0,  # "beam" -> "undefined"
    10: 3,  # "pillar" -> "wall"
    11: 6,  # "pole"
    12: 0,  # "steps" -> "undefined"
    13: 0,  # "lamp" -> "undefined"
    14: 1,  # "rail_tie" -> "ground"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess subway dataset")
    parser.add_argument("--dataset_root", required=True, help="Path to raw LAZ files")
    parser.add_argument("--output_root", required=True, help="Path to processed dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for creating dataset splits")
    parser.add_argument("--k_neighbors", type=int, default=30, help="Number of neighbors for normal estimation")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for processing")
    return parser.parse_args()


def init_worker():
    """Initialize worker process to ignore SIGINT"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def downsample_points(coords, voxel_size: float):
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(coords)
    pcd_downsampled = pcd_temp.voxel_down_sample(voxel_size=voxel_size)
    downsampled_coords = np.asarray(pcd_downsampled.points).astype(np.float32)

    # Map attributes from original to downsampled points
    tree = cKDTree(coords)
    distances, indices = tree.query(downsampled_coords, k=1)
    return downsampled_coords, indices


def process_laz_file(laz_path, dataset_type: str, k_neighbors=30):
    assert dataset_type in ("synth", "real"), "dataset_type must be 'synth' or 'real'"

    input_las = laspy.read(laz_path)

    # Extract coordinates
    coords = np.vstack([input_las.x, input_las.y, input_las.z]).T.astype(np.float32)

    # Extract intensity
    if dataset_type == "synth":
        intensity = input_las.heliosAmplitude.astype(np.float32)
        intensity = (intensity / 2**32) * 255  # Normalize to [0, 255] range as expected by pointcept transforms and models.
    elif dataset_type == "real":
        intensity = input_las.intensity.astype(np.float32)
        intensity = (intensity / 2**16) * 255  # Normalize to [0, 255] range as expected by pointcept transforms and models.

    # Extract labels
    segment = input_las.label.astype(np.int32)

    if dataset_type == "synth":
        segment = np.vectorize(ORIGINAL_TO_REDUCED_CLASS_ID_MAPPING.get)(segment).astype(np.int32)
        instance = input_las.hitObjectId.astype(np.int32)
    elif dataset_type == "real":
        # Ensure segment is a contiguous array to avoid data corruption
        segment = np.ascontiguousarray(segment, dtype=np.int32)
        # Real has no instance annotations
        instance = np.zeros(segment.shape[0], dtype=np.int32)

    # Estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals).astype(np.float32)

    # Create output las object
    output_las = laspy.LasData(laspy.LasHeader(point_format=6, version="1.4"))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.int32))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32))
    output_las.x = coords[:, 0]
    output_las.y = coords[:, 1]
    output_las.z = coords[:, 2]
    output_las.intensity = intensity
    output_las.nx = normals[:, 0]
    output_las.ny = normals[:, 1]
    output_las.nz = normals[:, 2]
    output_las.label = segment.astype(np.int32)
    output_las.instance = instance.astype(np.int32)

    # Aggregate data for dataset statistics
    unique, counts = np.unique(segment, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_points = len(segment)

    return output_las, class_counts, total_points


def save_scene_laz(preprocessed_las, output_dir, scene_name):
    """
    Save processed scene data as LAZ file with remapped labels
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}.las")
    preprocessed_las.write(output_path)


def create_splits(scene_names, train_ratio, val_ratio, test_ratio, dataset_type: str, seed=42):
    """
    Create train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    random.seed(seed)
    scenes = scene_names.copy()
    random.shuffle(scenes)

    n_total = len(scenes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val :]

    return {f"{dataset_type}_train": train_scenes, f"{dataset_type}_val": val_scenes, f"{dataset_type}_test": test_scenes}


def process_scene_worker(args):
    """Worker function for processing and saving a single scene"""
    laz_file, split_name, output_root, dataset_type, k_neighbors = args
    try:
        scene_name = laz_file.stem
        split_dir = output_root / split_name

        preprocessed_las, class_counts, total_points = process_laz_file(str(laz_file), dataset_type, k_neighbors=k_neighbors)
        save_scene_laz(preprocessed_las, str(split_dir), scene_name)

        del preprocessed_las  # Free memory immediately
        return class_counts, total_points
    except Exception as e:
        raise Exception(f"Error processing {laz_file}: {str(e)}")


def process_and_save_scenes(laz_files, splits, output_root, dataset_type, k_neighbors=30, num_workers=1):
    print("=== Processing and saving scenes ===")

    # Create a mapping from scene name to split
    scene_to_split = {}
    for split_name, scenes in splits.items():
        for scene in scenes:
            scene_to_split[scene] = split_name

    # Create LAZ output directories
    for split_name in splits.keys():
        laz_split_dir = output_root / split_name
        laz_split_dir.mkdir(exist_ok=True)

    class_counts = {i: 0 for i in range(min(REDUCED_CLASS_ID_TO_NAME.keys()), max(REDUCED_CLASS_ID_TO_NAME.keys()) + 1)}
    num_total_points = 0

    # Multi-processing execution
    # Prepare arguments for workers
    worker_args = []
    for laz_file in laz_files:
        scene_name = laz_file.stem
        split_name = scene_to_split[scene_name]
        worker_args.append((laz_file, split_name, output_root, dataset_type, k_neighbors))

    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=init_worker) as pool:
            results = list(tqdm(pool.imap(process_scene_worker, worker_args), total=len(worker_args), desc="Processing and saving scenes"))

            # Aggregate results
            for file_class_counts, file_total in results:
                for class_id, cnt in file_class_counts.items():
                    class_counts[class_id] += cnt
                num_total_points += file_total

    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating workers...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    except Exception as e:
        traceback.print_exc()
        pool.terminate()
        pool.join()
        sys.exit(1)

    print(f"Total points: {num_total_points:,}")
    print(f"Number of classes: {len(class_counts)}")
    print("")
    print("Class distribution:")

    # Sort by class ID
    for class_id, count in class_counts.items():
        class_name = REDUCED_CLASS_ID_TO_NAME[class_id]
        percentage = (count / num_total_points) * 100
        print(f"  Class {class_id:3d} {class_name:<25}: {count:10,} points ({percentage:6.2f}%)")

    return class_counts, num_total_points


def main(dataset_type: str):
    args = parse_args()

    # Validate num_workers
    if args.num_workers < 1:
        args.num_workers = 1
    elif args.num_workers > cpu_count():
        print(f"Warning: num_workers ({args.num_workers}) exceeds CPU count ({cpu_count()}). Using {cpu_count()} workers.")
        args.num_workers = cpu_count()

    print(f"Using {args.num_workers} worker(s) for processing")

    # Create output directories
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in [f"{dataset_type}_train", f"{dataset_type}_val", f"{dataset_type}_test"]:
        (output_root / split).mkdir(exist_ok=True)

    # Get all LAZ files
    laz_files = sorted(list(Path(args.dataset_root).glob("*.laz")))

    if not laz_files:
        raise ValueError(f"No LAZ files found in {args.dataset_root}")

    print(f"Found {len(laz_files)} point cloud files")

    scene_names = [laz_file.stem for laz_file in laz_files]

    print("=== Split statistics ===")
    splits = create_splits(scene_names, args.train_ratio, args.val_ratio, args.test_ratio, dataset_type, args.seed)

    print(f"Train: {len(splits[f'{dataset_type}_train'])} scenes")
    print(f"Val: {len(splits[f'{dataset_type}_val'])} scenes")
    print(f"Test: {len(splits[f'{dataset_type}_test'])} scenes")
    print("")

    sorted_class_counts, num_total_points = process_and_save_scenes(
        laz_files, splits, output_root, dataset_type, k_neighbors=args.k_neighbors, num_workers=args.num_workers
    )

    dataset_info = {
        "dataset_name": f"{dataset_type}_subway",
        "num_classes": len(sorted_class_counts),
        "classes": {id: {"name": name, "num_points": int(sorted_class_counts[id])} for id, name in REDUCED_CLASS_ID_TO_NAME.items()},
        "num_total_points": num_total_points,
        "splits": splits,
    }

    # Save split information
    with open(output_root / f"{dataset_type}_dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)

    print("")
    print("Dataset preprocessing complete!")
    print(f"Processed data saved to: {args.output_root}")


if __name__ == "__main__":
    main("synth")
