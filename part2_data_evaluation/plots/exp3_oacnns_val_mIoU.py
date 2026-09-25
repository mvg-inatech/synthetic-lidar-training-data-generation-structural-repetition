from common import make_plot


if __name__ == "__main__":
    make_plot(
        csv_filename="exp3_oacnns/input/val_mIoU.csv",
        out_filename_base="exp3_oacnns_val_mIoU",
        x_axis_column="Epoch",
        y_label="Validation mIoU [%]",
        legend_mapping={
            "subway/exp3_oacnns_tune_lr_0.0005 - val/mIoU": "5e-4",
            "subway/exp3_oacnns_tune_lr_0.001 - val/mIoU": "1e-3",
            "subway/exp3_oacnns_tune_lr_0.002 - val/mIoU": "2e-3",
            "subway/exp3_oacnns_tune_lr_0.003 - val/mIoU": "3e-3",
            "subway/exp3_oacnns_tune_lr_0.004 - val/mIoU": "4e-3",
            "subway/exp3_oacnns_tune_lr_0.005 - val/mIoU": "5e-3",
        },
        drawing_order=[
            "subway/exp3_oacnns_tune_lr_0.003 - val/mIoU",
            "subway/exp3_oacnns_tune_lr_0.004 - val/mIoU",
            "subway/exp3_oacnns_tune_lr_0.005 - val/mIoU",
            "subway/exp3_oacnns_tune_lr_0.002 - val/mIoU",
            "subway/exp3_oacnns_tune_lr_0.0005 - val/mIoU",
            "subway/exp3_oacnns_tune_lr_0.001 - val/mIoU",
        ],
        figsize_cm=(12, 8),
        y_axis_factor=100,  # convert 0-1 to percentage
    )
