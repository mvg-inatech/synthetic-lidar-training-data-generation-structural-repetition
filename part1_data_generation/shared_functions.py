import os


def abort_if_output_directory_not_empty(output_path):
    if os.path.exists(output_path) and os.listdir(output_path):
        raise ValueError(f"Output directory '{output_path}' is not empty. Please clear it before running the script.")


def ensure_directory_exists(path):
    """Ensure that the directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)
    return path
