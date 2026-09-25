import multiprocessing
import numpy as np
import laspy
from pathlib import Path
from typing import List, Optional, Dict
import logging
import argparse
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET
from copy import deepcopy
from tqdm import tqdm
from class_info import get_class_name_to_id_mapping

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_scene_object_id_to_class_name_mapping(scene_path: Path) -> Dict[int, str]:
    """
    Load the scene XML file and create a mapping from object ID to semantic class.

    Args:
        scene_path: Path to the scene directory

    Returns:
        Dictionary mapping object IDs to semantic class names
    """
    xml_file = scene_path / "helios_input" / "scenes" / "blender2heliosScene.xml"

    if not xml_file.exists():
        raise FileNotFoundError(f"Scene XML file {xml_file} does not exist")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    scene_object_id_to_class_name_mapping = {}

    # Find all part elements and extract id and semantic_class
    for part in root.findall(".//part"):
        part_id = int(part.get("id"))
        semantic_class = part.get("semantic_class")
        scene_object_id_to_class_name_mapping[part_id] = semantic_class

    return scene_object_id_to_class_name_mapping


def find_scenes(base_path: str) -> List[str]:
    """
    Find all scene directories in the helios_workspace/helios_surveys folder.

    Args:
        base_path: Path to helios_workspace/helios_surveys

    Returns:
        List of scene names (directory names)
    """
    surveys_path = Path(base_path)
    if not surveys_path.exists():
        raise FileNotFoundError(f"Directory {base_path} does not exist")

    scenes = [d.name for d in surveys_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(scenes)} scenes")
    return scenes


def find_timestamp_directory(scene_path: Path) -> Path:
    """
    Find the unique timestamp directory within helios_output/sim.

    Args:
        scene_path: Path to the scene directory

    Returns:
        Path to the timestamp directory

    Raises:
        ValueError: If zero or multiple timestamp directories found
    """
    sim_path = Path(f"{scene_path}/helios_output/sim")
    if not sim_path.exists():
        raise FileNotFoundError(f"Directory {sim_path} does not exist")

    timestamp_dirs = [d for d in sim_path.iterdir() if d.is_dir()]

    if len(timestamp_dirs) == 0:
        raise ValueError(f"No timestamp directories found in {sim_path}")
    elif len(timestamp_dirs) > 1:
        raise ValueError(f"Multiple timestamp directories found in {sim_path}: {[d.name for d in timestamp_dirs]}")

    return timestamp_dirs[0]


def find_laz_file(timestamp_dir: Path) -> Path:
    """
    Find the unique .laz file in the timestamp directory.

    Args:
        timestamp_dir: Path to the timestamp directory

    Returns:
        Path to the .laz file

    Raises:
        ValueError: If zero or multiple .laz files found
    """
    laz_files = list(timestamp_dir.glob("*.laz"))

    if len(laz_files) == 0:
        raise ValueError(f"No .laz files found in {timestamp_dir}")
    elif len(laz_files) > 1:
        raise ValueError(f"Multiple .laz files found in {timestamp_dir}: {[f.name for f in laz_files]}")

    return laz_files[0]


def load_trajectory(trajectory_file: Path) -> np.ndarray:
    """
    Load tunnel trajectory from CSV file.

    Args:
        trajectory_file: Path to leg000_trajectory.txt

    Returns:
        Array of shape (n_points, 3) containing x,y,z coordinates
    """
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file {trajectory_file} does not exist")

    trajectory = np.loadtxt(trajectory_file, delimiter=" ", usecols=(0, 1, 2))
    logger.info(f"Loaded trajectory with {len(trajectory)} points ({trajectory_file})")
    return trajectory


def load_point_cloud(laz_file: Path) -> laspy.LasData:
    """
    Load point cloud from LAZ file.

    Args:
        laz_file: Path to the .laz file

    Returns:
        Loaded LAS data
    """
    if not laz_file.exists():
        raise FileNotFoundError(f"Point cloud file {laz_file} does not exist")

    las_data = laspy.read(laz_file)
    logger.info(f"Loaded point cloud with {len(las_data)} points")
    return las_data


def calculate_cumulative_distances(trajectory: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative distances along the trajectory path.

    Args:
        trajectory: Array of shape (n_points, 3) with trajectory coordinates

    Returns:
        Array of cumulative distances along the path
    """
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    # Calculate cumulative distances
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    return cumulative_distances


def generate_slice_boundaries(total_length: float, mean_length: float, std_length: float) -> List[float]:
    """
    Generate slice boundaries using Gaussian-sampled lengths.

    Args:
        total_length: Total length of the tunnel
        mean_length: Mean slice length in meters
        std_length: Standard deviation of slice length

    Returns:
        List of boundary positions along the tunnel
    """
    boundaries = [0.0]
    current_position = 0.0

    while current_position < total_length:
        slice_length = np.random.normal(mean_length, std_length)
        slice_length = max(slice_length, 1)  # Ensure minimum slice length

        current_position += slice_length
        if current_position >= total_length:
            boundaries.append(total_length)
            break
        else:
            boundaries.append(current_position)

    logger.info(f"Determined boundaries for {len(boundaries)} slices")
    return boundaries


def project_points_to_path(points: np.ndarray, trajectory: np.ndarray, cumulative_distances: np.ndarray) -> np.ndarray:
    """
    Project points onto the tunnel centerline and calculate distances along the path.
    Ultra-fast version using vectorized operations with spatial filtering.
    Pre-filters points to reduce search space.

    Args:
        points: Array of shape (n_points, 3) with point coordinates
        trajectory: Array of shape (n_traj_points, 3) with trajectory coordinates
        cumulative_distances: Cumulative distances along trajectory

    Returns:
        Array of distances along the path for each point
    """
    n_points = len(points)
    n_trajectory = len(trajectory)

    logger.info(f"Projecting {n_points} points to {n_trajectory} trajectory points")

    if n_trajectory == 0:
        raise ValueError("Trajectory is empty. Cannot build KD-tree.")

    # Build KD-tree for trajectory points
    tree = cKDTree(trajectory)

    # Prepare output
    path_distances = np.empty(n_points, dtype=float)

    # Process filtered points in chunks
    chunk_size = 500000  # Can be larger since we filtered
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk = points[start_idx:end_idx]

        # Find nearest trajectory points
        distances, indices = tree.query(chunk)

        # Map to cumulative distances
        path_distances[start_idx:end_idx] = cumulative_distances[indices]

        # if (i // chunk_size + 1) % 10 == 0:
        #    logger.info(f"Processed {end_idx}/{n_points} filtered points ({100 * end_idx / n_points:.1f}%)")

    return path_distances


def extract_slice_points(
    scene_las: laspy.LasData,
    point_path_distances: np.ndarray,
    start_distance: float,
    end_distance: float,
    scene_object_id_to_class_name_mapping: Dict[int, str],
    class_name_to_id: Dict[str, int],
) -> Optional[laspy.LasData]:
    """
    Extract points that fall within a specific slice along the path and add semantic class information.

    Args:
        scene_las: Original LAS data
        point_path_distances: Distance along path for each point
        start_distance: Start distance of the slice
        end_distance: End distance of the slice
        scene_object_id_to_class_name_mapping: Mapping from object ID to semantic class name
        class_name_to_id: Mapping from semantic class name to class ID

    Returns:
        New LAS data containing only points within the slice, or None if no points
    """
    # Find points within the slice boundaries
    mask = (point_path_distances >= start_distance) & (point_path_distances < end_distance)

    if not np.any(mask):
        logger.warning(f"No points found in slice [{start_distance:.2f}, {end_distance:.2f}]")
        return None

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="heliosAmplitude", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="hitObjectId", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int8))
    header.offsets = deepcopy(scene_las.header.offsets)
    header.scales = deepcopy(scene_las.header.scales)
    slice_las = laspy.LasData(header)

    # Copy all the original point dimensions for masked points
    for dimension_name in ["X", "Y", "Z", "intensity", "return_number", "number_of_returns", "gps_time", "hitObjectId", "heliosAmplitude"]:
        slice_las[dimension_name] = scene_las[dimension_name][mask]

    # Get the hitObjectId values for the sliced points
    points_hit_object_ids = scene_las.hitObjectId[mask]

    # Create class field based on semantic mapping
    points_class_ids = np.zeros(len(points_hit_object_ids), dtype=np.uint8)

    for point_idx, hit_obj_id in enumerate(points_hit_object_ids):
        hit_class_name = scene_object_id_to_class_name_mapping.get(hit_obj_id, "undefined")
        class_id = class_name_to_id[hit_class_name]
        points_class_ids[point_idx] = class_id

    slice_las["label"] = points_class_ids

    # logger.info(f"Extracted {len(slice_las)} points for slice [{start_distance:.2f}, {end_distance:.2f}]")
    return slice_las


def save_slice(slice_las: laspy.LasData, output_path: str) -> None:
    """
    Save a slice as a LAZ file.

    Args:
        slice_las: LAS data to save
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    slice_las.write(output_path)


def process_scene(scene_name: str, base_path: str, output_dir: str) -> None:
    """
    Process a single scene: load data, slice, and save.

    Args:
        scene_name: Name of the scene
        base_path: Base path to helios_workspace/helios_surveys
        output_dir: Output directory for sliced point clouds
    """
    logger.info(f"Processing scene: {scene_name}")

    scene_path = Path(base_path) / scene_name

    # Load semantic class mappings
    class_name_to_id = get_class_name_to_id_mapping()
    scene_object_id_to_class_name_mapping = get_scene_object_id_to_class_name_mapping(scene_path)

    # Find timestamp directory
    timestamp_dir = find_timestamp_directory(scene_path)

    # Find the unique .laz file and trajectory file
    laz_file = find_laz_file(timestamp_dir)
    trajectory_file = timestamp_dir / "leg000_trajectory.txt"

    trajectory = load_trajectory(trajectory_file)
    las_data = load_point_cloud(laz_file)

    # Calculate path distances
    cumulative_distances = calculate_cumulative_distances(trajectory)
    total_length = cumulative_distances[-1]

    # Generate slice boundaries with respect to the noisy robot trajectory.
    # Note that a mean_length of 15 m does not lead to 15 m long output slices.
    # The length of the output slices will be shorter, because mean_length is wrt. the noisy robot trajectory (which includes side-shifts away from the main tunnel axis).
    # In practice, if mean_length is set to 15 m, the length of the output slices is roughly 9 m.
    # Some of the first and last tunnel slices (before the trajectory starts and after it ends) may have longer lengths, around 20 m.
    boundaries = generate_slice_boundaries(total_length, mean_length=15, std_length=1)

    # Project points to path
    points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
    point_path_distances = project_points_to_path(points, trajectory, cumulative_distances)

    # Process each slice
    output_path = Path(output_dir)
    slice_count = 0

    for i in range(len(boundaries) - 1):
        # logger.info(f"Processing slice {i + 1}/{len(boundaries) - 1}")
        start_dist = boundaries[i]
        end_dist = boundaries[i + 1]

        # Extract slice points with semantic class information
        slice_las = extract_slice_points(las_data, point_path_distances, start_dist, end_dist, scene_object_id_to_class_name_mapping, class_name_to_id)

        if slice_las is not None:
            # Generate output filename
            output_filename = f"synth_{scene_name}_part-{slice_count:03d}_OusterOS0LowRes.laz"
            output_file_path = f"{output_path}/{output_filename}"

            # Save slice
            save_slice(slice_las, output_file_path)
            slice_count += 1

    # logger.info(f"Successfully processed scene {scene_name}: {slice_count} slices created")


def process_scene_wrapper(params):
    scene_name, args = params
    try:
        seed = int(int.from_bytes(scene_name.encode("utf-8"), "little") / (2**32 - 1))
        seed_random_generators(seed)
        process_scene(scene_name, args.base_path, args.output_dir)
    except Exception:
        logger.exception(f"Error processing scene {scene_name}", exc_info=True)
        raise


def process_scenes_with_progress(scenes, args, max_workers=24):
    scenes = list(scenes)  # ensure we can get total for the progress bar
    total = len(scenes)
    if total == 0:
        logger.info("No scenes to process.")
        return

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=max_workers)

    try:
        work_iter = ((scene, args) for scene in scenes)
        it = pool.imap_unordered(process_scene_wrapper, work_iter, chunksize=1)  # chunksize=1 for per-item progress updates

        with tqdm(total=total, desc="Processing scenes", unit="scene") as pbar:
            for _ in it:
                pbar.update(1)

        pool.close()
        pool.join()
        logger.info("All scenes processing complete!")
    except Exception:
        # Fail fast: kill all workers immediately, then re-raise
        pool.terminate()
        pool.join()
        raise


def main() -> None:
    """
    Main function to process scenes based on command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process point cloud scenes and slice them into segments")
    parser.add_argument("--scene", type=str, help="Name of a single scene to process. If not provided, all scenes will be processed.", default=None)
    parser.add_argument("--base-path", type=str, help="Path to helios_workspace/helios_surveys", default="helios_workspace/helios_surveys")
    parser.add_argument("--output-dir", type=str, help="Output directory for sliced point clouds", default="sliced_tunnels")

    args = parser.parse_args()

    if args.scene:
        # Verify the scene exists
        scene_path = Path(args.base_path) / args.scene
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene directory {scene_path} does not exist")

        scenes = [args.scene]
    else:
        # Process all scenes
        scenes = sorted(find_scenes(args.base_path))

    if not scenes:
        raise ValueError("No scenes found")

    process_scenes_with_progress(scenes, args)

    logger.info("Processing complete!")


def seed_random_generators(seed: int):
    """
    Seed random number generators for reproducibility.
    """
    np.random.seed(seed)


if __name__ == "__main__":
    main()
