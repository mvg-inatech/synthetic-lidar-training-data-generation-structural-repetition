from common import make_plot


if __name__ == "__main__":
    make_plot(
        csv_filename="exp9_ptv3/input/val_mIoU.csv",
        out_filename_base="exp9_ptv3_val_mIoU",
        x_axis_column="Epoch",
        y_label="Validation mIoU [%]",
        legend_mapping={
            "subway/exp3_sonata_tune_lr_0.001 - val/mIoU": "(a) 3 channels, pretrained",
            "subway/exp9_sonata_no_color_init - val/mIoU": "(b) 3 channels, from scratch",
            "subway/exp9_sonata_no_color_init_1ch - val/mIoU": "(c) 1 channel, from scratch",
        },
        drawing_order=[
            "subway/exp3_sonata_tune_lr_0.001 - val/mIoU",
            "subway/exp9_sonata_no_color_init - val/mIoU",
            "subway/exp9_sonata_no_color_init_1ch - val/mIoU",
        ],
        figsize_cm=(12, 8),
        y_axis_factor=100,  # convert 0-1 to percentage
    )
