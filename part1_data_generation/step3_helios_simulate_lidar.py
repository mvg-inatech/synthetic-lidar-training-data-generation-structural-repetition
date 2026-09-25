import multiprocessing
import os
import signal
import time
from typing import List
import numpy as np
import pyhelios
from tqdm import tqdm
import json
import argparse
from scipy.interpolate import CubicSpline
from tunnel_type_trajectory_shift_mapping import TUNNEL_TYPE_TRAJECTORY_SHIFT_MAPPING


def get_trajectory_waypoints(generated_tunnels_blender_path: str, scene_name: str) -> np.array:
    metadata_file_path = os.path.join(generated_tunnels_blender_path, f"{scene_name}_metadata.json")
    with open(metadata_file_path, "r") as file:
        metadata = json.load(file)

    assert metadata["slice_type"] in TUNNEL_TYPE_TRAJECTORY_SHIFT_MAPPING.keys(), (
        f"Unknown tunnel slice type: {metadata['slice_type']} (you must define a trajectory shift)"
    )
    tunnel_type_shift = TUNNEL_TYPE_TRAJECTORY_SHIFT_MAPPING[metadata["slice_type"]]

    tunnel_center_waypoints = np.array(metadata["curve_points"])  # The spline points the tunnel slice is bent along

    # Add a shift for the robot trajectory.
    # This is especially relevant for the tunnel slice types that contain two tracks, for cases where the robot should not be walking in the middle of the tunnel slice, but rather on one of the two tracks.

    # Calculate local coordinate systems for each waypoint
    n_points = len(tunnel_center_waypoints)

    # Calculate tangent vectors (Y-axis of local coordinate system)
    tangents = np.zeros_like(tunnel_center_waypoints)

    # First point: forward difference
    tangents[0] = tunnel_center_waypoints[1] - tunnel_center_waypoints[0]

    # Middle points: central difference
    for i in range(1, n_points - 1):
        tangents[i] = tunnel_center_waypoints[i + 1] - tunnel_center_waypoints[i - 1]

    # Last point: same as second-last
    tangents[-1] = tangents[-2]

    # Normalize tangents
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    # Calculate up vectors (Z-axis of local coordinate system)
    up_vectors = np.zeros_like(tunnel_center_waypoints)

    # Initial up vector (global Z, since first slice points along global Y with global Z up)
    up_vectors[0] = np.array([0, 0, 1])

    # Transport up vector along the curve, keeping it perpendicular to tangent
    for i in range(1, n_points):
        # Start with previous up vector
        prev_up = up_vectors[i - 1]
        current_tangent = tangents[i]

        # Remove component parallel to tangent to ensure perpendicularity
        up_vectors[i] = prev_up - np.dot(prev_up, current_tangent) * current_tangent

        # Normalize (handle potential zero vector)
        norm = np.linalg.norm(up_vectors[i])
        if norm > 1e-8:
            up_vectors[i] = up_vectors[i] / norm
        else:
            # Fallback: create a vector perpendicular to tangent
            # Find a vector not parallel to current_tangent
            if abs(current_tangent[0]) < 0.9:
                temp = np.array([1, 0, 0])
            else:
                temp = np.array([0, 1, 0])
            up_vectors[i] = temp - np.dot(temp, current_tangent) * current_tangent
            up_vectors[i] = up_vectors[i] / np.linalg.norm(up_vectors[i])

    # Calculate radial vectors (X-axis of local coordinate system)
    # Using right-hand rule: X = Y × Z (tangent × up)
    radial_vectors = np.cross(tangents, up_vectors)

    # Apply shift in local coordinate system
    robot_trajectory_waypoints = tunnel_center_waypoints.copy()

    for i in range(n_points):
        # Apply shift in local coordinates:
        # shift_x: radial direction (perpendicular to tunnel axis)
        # shift_y: longitudinal direction (along tunnel axis)
        # shift_z: vertical direction (toward ceiling)
        local_shift = (
            tunnel_type_shift["shift_x"] * radial_vectors[i] + tunnel_type_shift["shift_y"] * tangents[i] + tunnel_type_shift["shift_z"] * up_vectors[i]
        )

        robot_trajectory_waypoints[i] += local_shift

    return robot_trajectory_waypoints


def interpolate_waypoints(waypoints, time_step):
    """Interpolates waypoints to create a smooth trajectory using cubic splines."""
    # Extract columns
    t = waypoints[:, 0]
    x = waypoints[:, 1]
    y = waypoints[:, 2]
    z = waypoints[:, 3]

    # Generate new time points
    new_t = np.arange(t[0], t[-1] + time_step, time_step)
    fx = CubicSpline(t, x)
    fy = CubicSpline(t, y)
    fz = CubicSpline(t, z)

    # Interpolate all dimensions
    new_x = fx(new_t)
    new_y = fy(new_t)
    new_z = fz(new_t)

    # Combine into output array
    return np.column_stack((new_t, new_x, new_y, new_z))


def compute_yaw_angles(traj):
    yaw_angles = []
    for i in range(len(traj) - 1):
        p1 = traj[i, 1:4]  # x, y, z
        p2 = traj[i + 1, 1:4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        yaw = np.arctan2(dx, dy)
        yaw = np.degrees(yaw)

        # Normalize yaw to be in the range [0, 360)
        if yaw < 0:
            yaw += 360

        yaw_angles.append(yaw)

    # Append the last yaw angle
    yaw_angles.append(yaw_angles[-1])

    return np.array(yaw_angles)


def add_noise_to_waypoints(waypoints):
    """Adds Gaussian noise to the waypoints with separate standard deviations for x,y,z."""
    # Standard deviations for noise in each dimension in meters
    noise_std_x = 0.15
    noise_std_y = 0.15
    noise_std_z = 0.05

    noise_x = np.random.normal(0, noise_std_x, len(waypoints))
    noise_y = np.random.normal(0, noise_std_y, len(waypoints))
    noise_z = np.random.normal(0, noise_std_z, len(waypoints))

    noisy_waypoints = waypoints.copy()
    noisy_waypoints[:, 1] += noise_x
    noisy_waypoints[:, 2] += noise_y
    noisy_waypoints[:, 3] += noise_z

    return noisy_waypoints


def generate_trajectory_csv(trajectory_csv_path: str, waypoints: np.array):
    # compute time assuming constant speed
    # This speed value is used for the platform. When helios++ runs, it will print that it uses another default speed value coming from the groundvehicle platform. This is ignored because we employ the interpolated platform using this speed value, and only referencing the groundvehicle platform as the `basePlatform` for the scanner mount, etc.
    speed_m_s = 3  # meters per second

    distances_between_waypoints = np.linalg.norm(np.diff(np.array(waypoints)[:, :3], axis=0), axis=1)
    times = [0]
    for distance in distances_between_waypoints:
        times.append(times[-1] + distance / speed_m_s)
    waypoints = np.column_stack((times, waypoints))

    # interpolate waypoints to create a trajectory
    time_step = 0.01  # seconds
    waypoints_interp = interpolate_waypoints(waypoints, time_step=time_step)
    waypoints_interp = add_noise_to_waypoints(waypoints_interp)
    roll = np.zeros(len(waypoints_interp))  # Assuming roll is zero for simplicity
    pitch = np.zeros(len(waypoints_interp))  # Assuming pitch is zero for simplicity
    yaw = compute_yaw_angles(waypoints_interp)
    waypoints_interp = np.column_stack((waypoints_interp, roll, pitch, yaw))

    np.savetxt(trajectory_csv_path, waypoints_interp, delimiter=",", fmt="%.6f")
    print(f"Generated trajectory CSV at {trajectory_csv_path}")


def run_helios_simulation(helios_config_path, helios_assets_path, helios_output_path, helios_seed, fast_mode: bool = False):
    start_time = time.time()

    # Sim context.
    # Set logging.
    # pyhelios.loggingQuiet()
    # pyhelios.loggingSilent()
    pyhelios.loggingDefault()
    # pyhelios.loggingVerbose()
    # pyhelios.loggingVerbose2()

    # Set seed for default random number generator.
    pyhelios.setDefaultRandomnessGeneratorSeed(helios_seed)

    # Build simulation parameters
    simBuilder = pyhelios.SimulationBuilder(helios_config_path, helios_assets_path, helios_output_path)
    simBuilder.setNumThreads(0)
    simBuilder.setLasOutput(True)
    simBuilder.setZipOutput(True)
    simBuilder.setCallbackFrequency(0)  # Run without callback
    simBuilder.setFinalOutput(False)
    simBuilder.setExportToFile(True)
    simBuilder.setRebuildScene(not fast_mode)

    sim = simBuilder.build().sim

    sim.start()

    # Create instance of PyHeliosOutputWrapper class using sim.join().
    # Returns attributes 'measurements' and 'trajectories' which are Python wrappers of classes that contain the output vectors.
    sim.join()

    simulation_duration = time.time() - start_time
    print(f"Simulation duration: {simulation_duration:.2f} seconds")


def generate_helios_config(scene_directory_path, helios_config_template_path):
    # Load the template config file
    with open(helios_config_template_path, "r") as file:
        helios_config_template = file.read()

    # Replace placeholders in the template with actual values
    helios_config = helios_config_template.replace("{{PLACEHOLDER_SCENE_DIRECTORY_PATH}}", scene_directory_path)

    # Write the modified config to a new file
    output_path = os.path.join(scene_directory_path, "helios_input", "helios_config.xml")
    with open(output_path, "w") as file:
        file.write(helios_config)

    print(f"Generated Helios config for {scene_directory_path} at {output_path}")


def get_signal_name_from_num(n: int) -> str:
    """
    Convert a signal number (may be negative, e.g. -9) to its name (e.g. 'SIGKILL').
    Falls back to 'SIG<NUM>' if the signal is unknown on this system.
    """
    if not isinstance(n, int):
        raise TypeError("signal number must be an int")

    if n < 0:
        n = -n

    try:
        return signal.Signals(n).name
    except Exception:
        # Fallback: search in signal module dict
        for name, val in getattr(signal, "__dict__", {}).items():
            if name.startswith("SIG") and isinstance(val, int) and val == n:
                return name
        return f"SIG{n}"


def filter_scenes_output_exists_already(scene_directory_names: List[str]) -> List[str]:
    missing_scenes = []
    for scene in scene_directory_names:
        output_dir = os.path.join("helios_workspace", "helios_surveys", scene, "helios_output", "sim")
        if os.path.exists(output_dir):
            print(f"Note: Skipping {scene} (output exists)")
        else:
            missing_scenes.append(scene)
    return missing_scenes


def parse_args():
    parser = argparse.ArgumentParser(description="Run HELIOS++ simulations for synthetic data generation.")
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        help="Do not rebuild existing scene files. Use this if you have already run the simulation before and only want to run it again with different parameters, without having changed the input data.",
    )
    parser.add_argument(
        "--skip_existing_output_scenes",
        action=argparse.BooleanOptionalAction,
        help="Skip scenes for which the output directory exists already (probably they have been simulated before).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Run the simulation for a specific scene. If unset, run for all scenes.",
    )
    parser.add_argument(
        "--start_at",
        type=int,
        help="Run the simulation starting from a specific seed. If unset, run from the beginning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.start_at is None or args.start_at >= 0, "--start_at must be a non-negative integer."
    assert (args.start_at is None) or (args.scene is None), "Cannot use --start_at together with --scene."

    helios_config_template_path = "helios_config_template.xml"
    base_path = os.path.join("helios_workspace", "helios_surveys")
    scene_directory_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    if args.scene:
        scene_directory_names = [args.scene] if args.scene in scene_directory_names else []

    if args.start_at is not None:
        scene_directory_names = [f"seed-{i}" for i in range(args.start_at, len(scene_directory_names))]

    if args.skip_existing_output_scenes:
        scene_directory_names = filter_scenes_output_exists_already(scene_directory_names)

    if not scene_directory_names:
        raise ValueError("No valid scenes found to process.")

    for scene_directory_name in tqdm(scene_directory_names):
        generate_helios_config(
            os.path.join(base_path, scene_directory_name),
            helios_config_template_path,
        )

        generate_trajectory_csv(
            os.path.join(base_path, scene_directory_name, "helios_input", "trajectory.csv"),
            get_trajectory_waypoints(
                os.path.join("helios_workspace", "generated_tunnels_blender"),
                scene_directory_name,
            ),
        )

        try:
            # Start simulation in a separate process to ensure memory is freed after each single simulation. This is a workaround for issue https://github.com/3dgeo-heidelberg/helios/issues/639
            process = multiprocessing.Process(
                target=run_helios_simulation,
                args=(
                    os.path.join(base_path, scene_directory_name, "helios_input", "helios_config.xml"),
                    os.path.join(base_path, scene_directory_name, "assets"),  # Doesn't exist and doesn't need to
                    os.path.join(base_path, scene_directory_name, "helios_output"),
                    scene_directory_name.split("/")[-1].split("-")[-1],  # Extract seed from directory name, e.g., "seed-0" -> 0
                    args.fast,
                ),
            )
            process.start()
            process.join()

            # Detect abnormal termination
            if process.exitcode != 0:
                signal_name = get_signal_name_from_num(process.exitcode)
                msg = f"HELIOS++ subprocess for scene '{scene_directory_name}' was terminated by signal {signal_name}."
                if signal_name == "SIGKILL":
                    msg += " This is likely due to an out-of-memory (OOM) error. Consider reducing scene complexity or increasing available memory."
                raise RuntimeError(msg)

        # Need to manually handle KeyboardInterrupt because HELIOS++ doesn't respond to SIGTERM...
        except KeyboardInterrupt:
            if process.is_alive():
                process.terminate()
            raise
