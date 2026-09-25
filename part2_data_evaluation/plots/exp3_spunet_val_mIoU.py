from common import make_plot


if __name__ == "__main__":
    make_plot(
        csv_filename="exp3_spunet/input/val_mIoU.csv",
        out_filename_base="exp3_spunet_val_mIoU",
        x_axis_column="Epoch",
        y_label="Validation mIoU [%]",
        legend_mapping={
            "subway/exp3_spunet_tune_lr_0.0001 - val/mIoU": "1e-4",
            "subway/exp3_spunet_tune_lr_0.0005 - val/mIoU": "5e-4",
            "subway/exp3_spunet_tune_lr_0.001 - val/mIoU": "1e-3",
            "subway/exp3_spunet_tune_lr_0.0015 - val/mIoU": "15e-4",
            "subway/exp3_spunet_tune_lr_0.002 - val/mIoU": "2e-3",
            "subway/exp3_spunet_tune_lr_0.0025 - val/mIoU": "25e-4",
            "subway/exp3_spunet_tune_lr_0.05 - val/mIoU": "5e-2",
        },
        drawing_order=[
            "subway/exp3_spunet_tune_lr_0.0001 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.0005 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.001 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.0015 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.0025 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.05 - val/mIoU",
            "subway/exp3_spunet_tune_lr_0.002 - val/mIoU",
        ],
        figsize_cm=(12, 8),
        y_axis_factor=100,  # convert 0-1 to percentage
    )
