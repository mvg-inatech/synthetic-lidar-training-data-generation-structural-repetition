"""
Generate scientific plots from CSV files exported from wandb.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional, Dict
import numpy as np

# Configure matplotlib to use LaTeX-style fonts (Computer Modern-like) and 10pt font size
# This matches the elsarticle LaTeX template without requiring LaTeX installation
plt.rcParams.update(
    {
        "text.usetex": False,  # Don't require LaTeX installation
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "New Century Schoolbook", "Century Schoolbook L", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "mathtext.fontset": "cm",  # Use Computer Modern for math text
        "axes.formatter.use_mathtext": True,  # Use mathtext for scientific notation
    }
)


def get_csv_files(input_dir: Path) -> list[Path]:
    """
    Find all CSV files in the input directory.

    Args:
        input_dir: Path to the input directory

    Returns:
        List of Path objects for CSV files
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{input_dir}'")

    return csv_files


def load_csv_data(csv_file: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        csv_file: Path to the CSV file

    Returns:
        DataFrame containing the CSV data
    """
    df = pd.read_csv(csv_file)
    return df


def get_columns_to_plot(df: pd.DataFrame, x_axis_column: str) -> list[str]:
    """
    Filter columns to plot, excluding MIN/MAX suffixes and the x-axis column.

    Args:
        df: DataFrame with the data
        x_axis_column: Name of the x-axis column to exclude

    Returns:
        List of column names to plot
    """
    columns_to_plot = []
    for column_name in df.columns:
        if column_name == x_axis_column:
            continue
        if "__MIN" in column_name or "__MAX" in column_name or "_step" in column_name:
            continue
        columns_to_plot.append(column_name)

    return columns_to_plot


def get_colormap_colors(n_colors: int, colormap_name: str = "tab10", max_colors: int = 10) -> list:
    """
    Generate colors from a matplotlib colormap.

    Args:
        n_colors: Number of colors needed
        colormap_name: Name of the colormap to use
        max_colors: Number of distinct colors in the colormap

    Returns:
        List of RGBA color tuples
    """
    cmap = plt.get_cmap(colormap_name)
    return [cmap(i % max_colors) for i in range(n_colors)]


def apply_legend_mapping(column_name: str, legend_mapping: Optional[Dict[str, str]]) -> str:
    """
    Apply custom legend mapping if provided.

    Args:
        column_name: Original column name
        legend_mapping: Optional dictionary mapping column names to custom labels

    Returns:
        Mapped label or original column name
    """
    if legend_mapping and column_name in legend_mapping:
        return legend_mapping[column_name]
    return column_name


def calculate_nice_ticks(data_min: float, data_max: float, target_count: int = 5, integer_only: bool = False) -> np.ndarray:
    """
    Calculate nice tick values that cover the data range with round numbers.

    Args:
        data_min: Minimum data value
        data_max: Maximum data value
        target_count: Target number of ticks
        integer_only: If True, only return integer tick values

    Returns:
        Array of tick positions
    """
    data_range = data_max - data_min

    if data_range == 0:
        return np.array([data_min])

    # Calculate rough step size
    rough_step = data_range / (target_count - 1)

    # Find nice step size (1, 2, 5, 10, 20, 50, etc.)
    if integer_only:
        magnitude = 10 ** np.floor(np.log10(max(rough_step, 1)))
        nice_steps = [1, 2, 5, 10]
    else:
        magnitude = 10 ** np.floor(np.log10(rough_step))
        nice_steps = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10]

    normalized_steps = [s * magnitude for s in nice_steps]
    step = min(normalized_steps, key=lambda x: abs((data_range / x) - (target_count - 1)))

    # Start from a nice round number at or just below data_min
    if integer_only:
        tick_start = int(np.ceil(data_min / step) * step)
    else:
        tick_start = np.ceil(data_min / step) * step

    ticks = []
    current_tick = tick_start
    while current_tick <= data_max + step * 0.001:  # Small epsilon for float comparison
        ticks.append(current_tick)
        current_tick += step

    ticks = np.array(ticks)

    # Ensure we have at least 3 ticks
    if len(ticks) < 3:
        return np.linspace(data_min, data_max, target_count)

    return ticks


def create_line_plot(
    df: pd.DataFrame,
    x_column: str,
    columns_to_plot: list[str],
    colors: list,
    y_label: str,
    y_axis_factor: float = 1.0,
    legend_mapping: Optional[Dict[str, str]] = None,
    drawing_order: Optional[list[str]] = None,
    figsize_cm: tuple[float, float] = (15.0, 10.0),
) -> tuple[Figure, Axes]:
    """
    Create a line plot with multiple series.

    Args:
        df: DataFrame with the data
        x_column: Name of the column to use for x-axis
        columns_to_plot: List of column names to plot
        colors: List of colors to use for lines
        y_label: Label for the y-axis
        y_axis_factor: Factor to scale the y-axis values
        legend_mapping: Optional dictionary to map column names to custom legend labels.
            The legend order will follow the key order in this dictionary.
        drawing_order: Optional list specifying the order in which lines are drawn.
            Lines drawn later appear on top. Does not affect legend order.
        figsize_cm: Figure size as (width, height) in centimeters. Default is (15.0, 10.0)

    Returns:
        Tuple of (figure, axes) objects
    """
    # Convert cm to inches (1 inch = 2.54 cm)
    figsize_inches = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    fig, ax = plt.subplots(figsize=figsize_inches)

    # Reorder columns based on legend_mapping key order if provided
    if legend_mapping:
        ordered_columns = [col for col in legend_mapping.keys() if col in columns_to_plot]
        # Do not add any columns not in mapping at the end
        columns_to_plot = ordered_columns

    # Determine drawing order (may differ from legend order)
    if drawing_order:
        draw_columns = [col for col in drawing_order if col in columns_to_plot]
        # Add any remaining columns not in drawing_order
        remaining = [col for col in columns_to_plot if col not in drawing_order]
        draw_columns = draw_columns + remaining
    else:
        draw_columns = columns_to_plot

    # Create a mapping from column to its index in legend order (for consistent colors)
    col_to_legend_idx = {col: idx for idx, col in enumerate(columns_to_plot)}

    # Draw lines in drawing order, storing handles for legend reordering
    handles_dict = {}
    for col in draw_columns:
        idx = col_to_legend_idx[col]
        color = colors[idx % len(colors)]
        label = apply_legend_mapping(col, legend_mapping)
        (line,) = ax.plot(df[x_column], df[col] * y_axis_factor, label=label, color=color, linewidth=2, marker="o", markersize=4)
        handles_dict[col] = line

    # Build legend in the correct order (legend_mapping order)
    legend_handles = [handles_dict[col] for col in columns_to_plot]
    legend_labels = [apply_legend_mapping(col, legend_mapping) for col in columns_to_plot]

    # Configure axes
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_label)
    ax.legend(legend_handles, legend_labels, loc="best")  # , framealpha=0.9, edgecolor="gray", fancybox=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Calculate data ranges
    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_values = df[columns_to_plot].values.flatten() * y_axis_factor
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)

    # Determine if x-axis should use integers (common for epochs, steps, etc.)
    x_is_integer = df[x_column].dtype in [np.int32, np.int64] or all(df[x_column] == df[x_column].astype(int))

    # Calculate beautiful ticks
    x_ticks = calculate_nice_ticks(x_min, x_max, target_count=6, integer_only=x_is_integer)
    y_ticks = calculate_nice_ticks(y_min, y_max, target_count=6, integer_only=True)

    # Add padding to y-axis (5% of the range on top and bottom)
    y_range = y_max - y_min
    y_padding = y_range * 0.05

    # Set axis limits to show full data range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Set the ticks
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Format tick labels nicely
    if not x_is_integer and x_ticks.max() < 10:
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    if y_ticks.max() < 10 and y_ticks.min() > -10:
        # Use appropriate precision based on the range
        y_range = y_ticks.max() - y_ticks.min()
        if y_range < 1:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        elif y_range < 10:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    plt.tight_layout()

    return fig, ax


def save_plot(fig: Figure, output_path: Path) -> None:
    """
    Save the plot to both PDF and PNG files.

    Args:
        fig: Figure object to save
        output_path: Path where to save the files (without extension)
    """
    # Save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"  Saved PDF to: {pdf_path.name}")

    # Save as PNG
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  Saved PNG to: {png_path.name}")


def process_csv_file(
    csv_file: Path,
    output_dir: Path,
    out_filename_base: str,
    x_column: str,
    y_label: str,
    y_axis_factor: float = 1.0,
    legend_mapping: Optional[Dict[str, str]] = None,
    drawing_order: Optional[list[str]] = None,
    figsize_cm: tuple[float, float] = (15.0, 10.0),
) -> None:
    """
    Process a single CSV file and generate a plot.

    Args:
        csv_file: Path to the CSV file
        output_dir: Directory where to save the output
        out_filename_base: Base name for the output files
        x_column: Name of the column to use for x-axis
        y_label: Label for the y-axis
        y_axis_factor: Factor to scale the y-axis values
        legend_mapping: Optional dictionary to map column names to custom legend labels
        drawing_order: Optional list specifying the order in which lines are drawn
        figsize_cm: Figure size as (width, height) in centimeters. Default is (15.0, 10.0)
    """
    print(f"\nProcessing: {csv_file.name}")

    df = load_csv_data(csv_file)

    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in {csv_file.name}")

    columns_to_plot = get_columns_to_plot(df, x_column)
    colors = get_colormap_colors(len(columns_to_plot))

    fig, _ = create_line_plot(df, x_column, columns_to_plot, colors, y_label, y_axis_factor, legend_mapping, drawing_order, figsize_cm)

    output_path = output_dir / "output"
    output_path.mkdir(exist_ok=True)
    output_path = output_path / out_filename_base
    save_plot(fig, output_path)

    plt.close(fig)


def make_plot(
    csv_filename: str,
    out_filename_base: str,
    y_label: str,
    y_axis_factor: float = 1.0,
    x_axis_column: str = "Epoch",
    legend_mapping: Optional[Dict[str, str]] = None,
    drawing_order: Optional[list[str]] = None,
    figsize_cm: tuple[float, float] = (15.0, 10.0),
) -> None:
    """
    Process a specific CSV file from wandb export and create a line plot.

    Args:
        csv_filename: Name of the CSV file to process (e.g., "data.csv")
        out_filename_base: Base name for the output files
        y_label: Label for the y-axis
        x_axis_column: Name of the column to use for x-axis
        legend_mapping: Optional dictionary to map CSV column names (keys) to custom legend labels (values)
        drawing_order: Optional list specifying the order in which lines are drawn (later = on top)
        figsize_cm: Figure size as (width, height) in centimeters. Default is (15.0, 10.0)
    """
    csv_file = Path(csv_filename)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist")

    output_dir = csv_file.parent.parent

    print(f"Processing CSV file: {csv_filename}")
    process_csv_file(csv_file, output_dir, out_filename_base, x_axis_column, y_label, y_axis_factor, legend_mapping, drawing_order, figsize_cm)

    print("\nPlot generated successfully!")
