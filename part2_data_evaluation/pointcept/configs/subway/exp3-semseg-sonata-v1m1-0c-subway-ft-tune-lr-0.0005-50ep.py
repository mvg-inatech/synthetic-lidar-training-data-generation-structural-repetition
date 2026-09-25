# Config is taken from pointcept repo, scannet semseg-sonata-v1m1-0c-scannet-ft, modified

_base_ = ["../_base_/default_runtime.py"]

dataset_type = "SubwayDataset"
data_root = "data/subway"

# This list must be hardcoded since this module is loaded by yapf and it only expects data structures but no functions that could load a json file from disk.
# The key defines the class ID, the value defines the class name.
CLASS_ID_TO_NAME = {
    0: "undefined",
    1: "ground",
    2: "rail",
    3: "wall",
    4: "power_rail",
    5: "power_rail_mount",
    6: "pole",
}
class_names_list = [v for v in CLASS_ID_TO_NAME.values()]
ignore_index = -100  # This value is not used in the subway dataset, but the variable must be set to some value.
# We need to include the undefined/clutter class in our benchmark, because unlucky cropping can lead to all points being labeled as undefined/clutter, which causes the loss to become NaN.

seed = 123


# misc custom setting
batch_size = 16  # bs: total bs in all gpus
num_worker = 32
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=len(class_names_list),
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
    freeze_backbone=False,
)

epoch = 50
eval_epoch = 50  # Make this equal to `epoch`, otherwise the dataset will be looped (see defaults.py)

# scheduler settings
base_lr = 0.0005
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr * 0.1],  # different lr for each param group
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# param group 1: default params
# param group 2: params that contain "block" in their names (the transformer blocks), with lr reduced (following the "block lr scaler" in the PTv3 paper)
param_dicts = [dict(keyword="block", lr=base_lr * 0.1)]

# dataset settings
dataset_type = "SubwayDataset"
data_root = "data/subway"

data = dict(
    num_classes=len(class_names_list),
    ignore_index=ignore_index,
    names=class_names_list,
    train=dict(
        type=dataset_type,
        split="synth_train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.4, dropout_application_ratio=1.0),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # RandomShift to help the model better differentiate floor and ceiling when testing synth-trained model on real data:
            dict(type="RandomShift", shift=((0, 0), (0, 0), (-1, 1))),
            dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02), # Original pointcept values
            dict(type="RandomJitter", sigma=0.01, clip=0.05),  # Added to increase robustness when testing synth-trained model on real data
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="synth_val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="synth_test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
                # This dummy augmentation must be present to disable test time augmentation. See pointcept README.md. Otherwise, pointcept will report all test metrics as 0.
            ],
        ),
    ),
)


# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    # dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),  # Test model_best checkpoint instead of model_last
]
