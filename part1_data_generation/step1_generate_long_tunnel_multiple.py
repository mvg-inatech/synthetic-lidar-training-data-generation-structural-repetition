from tqdm import tqdm

from shared_functions import abort_if_output_directory_not_empty, ensure_directory_exists
from step1_generate_long_tunnel_single import generate_long_tunnel


if __name__ == "__main__":
    tunnel_types_collection_names = [
        "tunnel slice - type 1 - base",
        "tunnel slice - type 1 - emergency exit",
        "tunnel slice - type 2 - base",
        "tunnel slice - type 3 - base",
        "tunnel slice - type 3 - emergency exit",
        "tunnel slice - type 4 - base",
        "tunnel slice - type 4 - emergency exit",
        "tunnel slice - type 5 - base",
        "tunnel slice - type 5 - emergency exit",
        "tunnel slice - type 6 - base",
        "tunnel slice - type 6 - emergency exit",
        "tunnel slice - type 7 - base",
        "tunnel slice - type 7 - emergency exit",
        "tunnel slice - type 8 - base",
        "tunnel slice - type 9 - base",
        "tunnel slice - type 10 - base",
        "tunnel slice - type 10 - emergency exit",
        "tunnel slice - type 11 - base",
        "tunnel slice - type 11 - emergency exit",
        "tunnel slice - type 12 - base",
        "tunnel slice - type 13 - base",
        "tunnel slice - type 14 - base",
        "tunnel slice - type 15 - base",
        # Over-represent the following type because it has a more different appearance than the others and PTv3 performed worse on this tunnel type in initial tests
        "tunnel slice - type 16 - base",
        "tunnel slice - type 16 - base",
        "tunnel slice - type 16 - base",
    ]

    max_seed = len(tunnel_types_collection_names) * 10  # Number of long tunnels to generate
    input_tunnel_slice_path = "helios_workspace/10 m slices (step 3 - split instances).blend"

    if max_seed < len(tunnel_types_collection_names):
        print("WARNING: max_seed should be greater than the number of tunnel types to ensure dataset diversity.")

    output_path = "helios_workspace/generated_tunnels_blender"
    ensure_directory_exists(output_path)
    abort_if_output_directory_not_empty(output_path)

    param_combinations = []

    for seed in range(max_seed):
        tunnel_type_collection_name = tunnel_types_collection_names[seed % len(tunnel_types_collection_names)]
        param_combinations.append(
            {
                "input_tunnel_slice_collection_name": tunnel_type_collection_name,
                "output_long_tunnel_save_path": f"{output_path}/seed-{seed}.blend",
                "seed": seed,
            }
        )

    for param_combination in tqdm(param_combinations):
        generate_long_tunnel(
            input_tunnel_slice_path=input_tunnel_slice_path,
            input_tunnel_slice_collection_name=param_combination["input_tunnel_slice_collection_name"],
            output_long_tunnel_save_path=param_combination["output_long_tunnel_save_path"],
            rng_seed=param_combination["seed"],
            num_segments=20,  # 25 leads to OOM for seed-13 and seed-37
        )

    print("All long tunnels generated successfully.")
