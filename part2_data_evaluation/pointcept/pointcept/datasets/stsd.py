import glob
import os
from collections.abc import Sequence
import laspy
import numpy as np
from pointcept.datasets.defaults import DefaultDataset
from pointcept.utils.logger import get_logger
from .builder import DATASETS


@DATASETS.register_module()
class STSDDataset(DefaultDataset):
    """
    STSD dataset for semantic segmentation.

    The dataset should be preprocessed using preprocess_stsd.py
    and organized in the following structure:

    stsd/
    ├── train/
    │   ├── ....las
    │   └── ...
    ├── test/
    │   └── ...
    ├── dataset_info.json
    """

    def __init__(self, data_fraction: float = 1.0, **kwargs):
        assert data_fraction > 0.0 and data_fraction <= 1.0, "data_fraction must be in (0.0, 1.0]"
        self.data_fraction = data_fraction
        self.logger = get_logger(__name__)
        super().__init__(**kwargs)

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.las"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.las"))
        else:
            raise NotImplementedError(f"Unsupported split type: {type(self.split)}")

        # Ensure deterministic file list order before determining the items to use, as glob does not guarantee order.
        data_list = sorted(data_list)

        num_items_to_use = int(len(data_list) * self.data_fraction)
        self.logger.info(f"Found {len(data_list)} files for split '{self.split}'. Using fraction {self.data_fraction} => {num_items_to_use} files.")
        return data_list[:num_items_to_use]

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        # self.logger.info(f"Loading item {data_path}")

        assert not self.cache, "Caching is not implemented"

        las = laspy.read(data_path)

        # Extract data from LAZ
        coord = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        intensity = las.intensity.astype(np.float32).reshape(-1, 1)
        normal = np.vstack([las.nx, las.ny, las.nz]).T.astype(np.float32)
        scene_id = os.path.basename(data_path).replace(".las", "")

        segment = las.label.astype(np.int32)
        instance = las.instance.astype(np.int32)

        segment = segment.reshape(-1).astype(np.int32)
        assert not np.any(segment < 0), f"Warning: Found labels < 0 in {self.get_data_name(idx)}, min label: {segment.min()}"
        assert not np.any(segment > 11), f"Warning: Found labels > 11 in {self.get_data_name(idx)}, max label: {segment.max()}"

        instance = instance.reshape(-1).astype(np.int32)

        data_dict = dict(
            coord=coord,
            normal=normal,
            strength=intensity,
            color=np.concatenate([intensity, intensity, intensity], axis=-1),
            segment=segment,
            instance=instance,
            scene_id=scene_id,
            name=self.get_data_name(idx),
            # Note: the RandomDropout transform only drops out certain key indices that are hardcoded.
            # You can let it drop other keys as well if you pass the `index_valid_keys` key here.
        )

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]
