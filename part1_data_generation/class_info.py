import logging
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASS_ID_TO_NAME = {
    -1: "undefined",
    0: "ground",
    1: "wall",
    2: "rail",
    3: "rail_tie_area",
    4: "power_rail_mount",
    5: "power_rail",
    6: "cables",
    7: "wall_behind_cables",
    8: "emergency_exit_notch",
    9: "beam",
    10: "pillar",
    11: "pole",
    12: "steps",
    13: "lamp",
    14: "rail_tie",
}


def get_class_name_to_id_mapping() -> Dict[str, int]:
    """
    Create a mapping from semantic class strings to class IDs.

    Returns:
        Dictionary mapping semantic class names to integer IDs
    """

    # Create mapping with 1-based indexing (0 is reserved for unknown/background)
    class_name_to_id = {name: id for id, name in CLASS_ID_TO_NAME.items()}
    logger.info(f"Dataset semantic class mapping: {class_name_to_id}")
    return class_name_to_id


if __name__ == "__main__":
    get_class_name_to_id_mapping()
