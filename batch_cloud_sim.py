import rasterio as rio
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import satellite_cloud_generator as scg
from rasterio.windows import Window


def process_tiff_with_clouds(input_path, output_folder, window_size=512, window_index=0):
    """
    Process a TIFF file from l1_toa folder and generate a new image with synthetic clouds.

    Args:
        input_path (str): Path to the input TIFF file
        output_folder (str): Path to save the output files
        window_size (int): Size of the window to process (default: 512)
        window_index (int): Index of the window to use (0, 1, or 2)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the base filename without extension
    base_filename = Path(input_path).stem

    # Define window positions - three different locations
    window_positions = [
        {"col_off": 300, "row_off": 600},  # Original position
        {"col_off": 800, "row_off": 400},  # A different position
        {"col_off": 500, "row_off": 1000}  # Another different position
    ]

    # Choose the window position based on the index
    position = window_positions[window_index % len(window_positions)]

    # Read the input TIFF file
    with rio.open(input_path) as src:
        # Get metadata
        meta = src.meta
        # Define window for processing
        window = Window(col_off=position["col_off"],
                        row_off=position["row_off"],
                        width=window_size,
                        height=window_size)
        # Read data within the window and normalize
        data = src.read(window=window) / 1e4
        transform = src.window_transform(window)

    # Convert to torch tensor for processing
    data_clean = torch.from_numpy(data)

    # Generate synthetic cloud
    cl, cmask = scg.add_cloud(
        data_clean,
        max_lvl=(0.95, 1.0),
        min_lvl=(0.0, 0.05),
        locality_degree=3,
        channel_offset=2,
        blur_scaling=2.0,
        cloud_color=True,
        return_cloud=True
    )

    # Generate segmentation mask (0: clear sky, 1: cloud, 2: thin cloud)
    seg = scg.segmentation_mask(cmask, thin_range=(0.1, 0.5))[0]

    cloud_meta = meta.copy()
    cloud_meta.update({
        'width': window_size,
        'height': window_size,
        'count': cl.shape[1],  # Number of channels in cloud data
        'transform': transform
    })

    # Update metadata for saving the original image
    original_meta = meta.copy()
    original_meta.update({
        'width': window_size,
        'height': window_size,
        'count': data.shape[0],  # Number of channels in original data
        'transform': transform
    })

    # Define output file paths with window index
    cloud_tiff_path = os.path.join(output_folder, f"{base_filename}_simuclouds_window{window_index}.tif")
    mask_png_path = os.path.join(output_folder, f"{base_filename}_simuclouds_mask_window{window_index}.png")
    original_tiff_path = os.path.join(output_folder, f"{base_filename}_original_window{window_index}.tif")

    # Save cloudy image as TIFF
    with rio.open(cloud_tiff_path, 'w', **cloud_meta) as dst:
        cloud_data = (cl[0].numpy() * 1e4).astype(cloud_meta['dtype'])
        dst.write(cloud_data)

    # Save original image as TIFF
    with rio.open(original_tiff_path, 'w', **original_meta) as dst:
        original_data = (data_clean.numpy() * 1e4).astype(original_meta['dtype'])
        dst.write(original_data)

    # Save mask as PNG
    mask = seg.numpy()
    mask_png = np.zeros_like(mask, dtype=np.uint8)
    mask_png[mask == 2] = 120  # thin cloud
    mask_png[mask == 1] = 255  # thick cloud

    Image.fromarray(mask_png).save(mask_png_path)
    print(f"Processed window {window_index} from {input_path} and saved outputs to {output_folder}")

    return cloud_tiff_path, mask_png_path, original_tiff_path


def process_all_tiffs(input_folder, output_base_folder):
    """
    Process all TIFF files in the input folder and its subfolders.

    Args:
        input_folder (str): Path to the input folder containing TIFF files
        output_base_folder (str): Base path for output folders
    """
    # Find all TIFF files in the input folder and its subfolders
    input_path = Path(input_folder)
    tiff_files = list(input_path.glob("**/*.tif"))

    print(f"Found {len(tiff_files)} TIFF files to process")

    for tiff_file in tiff_files:
        # Get the relative path of the file to preserve folder structure
        rel_path = tiff_file.relative_to(input_path)
        parent_folder = rel_path.parent

        # Create corresponding output folder
        output_folder = Path(output_base_folder) / parent_folder

        # Process the TIFF file with three different windows
        for window_idx in range(3):
            cloud_path, mask_path, original_path = process_tiff_with_clouds(
                str(tiff_file),
                str(output_folder),
                window_index=window_idx
            )

            print(f"Generated: {cloud_path}, {mask_path}, and {original_path}")


if __name__ == "__main__":
    # Define input and output folders
    l1_toa_folder = "/media/thomas/Arctus_data2/0_Arctus_Project/Ocean_Hackaton/data/l1_toa"
    output_folder = "/media/thomas/Arctus_data2/0_Arctus_Project/Ocean_Hackaton/data/synthetic_clouds"

    # Process all TIFF files
    process_all_tiffs(l1_toa_folder, output_folder)
