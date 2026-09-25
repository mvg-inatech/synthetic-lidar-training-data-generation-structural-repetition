from common import make_plot


if __name__ == "__main__":
    make_plot(
        csv_filename="exp2_ptv3/input/val_mIoU.csv",
        out_filename_base="exp2_ptv3_val_mIoU",
        x_axis_column="Epoch",
        y_label="Validation mIoU [%]",
        legend_mapping={
            "subway/exp2_sonata_simple_training_pretrained_ft - val/mIoU": "Pretrained",
            "subway/exp2_sonata_simple_training_from_scratch_ft - val/mIoU": "From scratch",
        },
        figsize_cm=(12, 8),
        y_axis_factor=100,  # convert 0-1 to percentage
    )
