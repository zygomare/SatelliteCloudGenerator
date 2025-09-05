import rasterio as rio
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import satellite_cloud_generator as scg
from rasterio.windows import Window


def process_tiff_with_clouds(input_path, output_folder, window_size=512):
    """
    Process a TIFF file from l1_toa folder and generate a new image with synthetic clouds.

    Args:
        input_path (str): Path to the input TIFF file
        output_folder (str): Path to save the output files
        window_size (int): Size of the window to process (default: 512)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the base filename without extension
    base_filename = Path(input_path).stem

    # Read the input TIFF file
    with rio.open(input_path) as src:
        # Get metadata
        meta = src.meta

        # Define window for processing (you can adjust this as needed)
        window = Window(col_off=300, row_off=600, width=window_size, height=window_size)

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

    # Update metadata for saving
    meta.update({
        'width': window_size,
        'height': window_size,
        'count': cl.shape[1],
        'transform': transform
    })

    # Define output file paths
    cloud_tiff_path = os.path.join(output_folder, f"{base_filename}_simuclouds.tif")
    mask_png_path = os.path.join(output_folder, f"{base_filename}_simuclouds_mask.png")

    # Save cloudy image as TIFF
    with rio.open(cloud_tiff_path, 'w', **meta) as dst:
        dst.write(cl[0].numpy())

    # Save mask as PNG
    mask = seg.numpy()
    mask_png = np.zeros_like(mask, dtype=np.uint8)
    mask_png[mask == 2] = 120  # thin cloud
    mask_png[mask == 1] = 255  # thick cloud

    Image.fromarray(mask_png).save(mask_png_path)

    print(f"Processed {input_path} and saved outputs to {output_folder}")

    return cloud_tiff_path, mask_png_path


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

        # Process the TIFF file
        cloud_path, mask_path = process_tiff_with_clouds(
            str(tiff_file),
            str(output_folder)
        )

        print(f"Generated: {cloud_path} and {mask_path}")


if __name__ == "__main__":
    # Define input and output folders
    l1_toa_folder = "/media/thomas/Arctus_data2/0_Arctus_Project/Ocean_Hackaton/data/l1_toa"
    output_folder = "/media/thomas/Arctus_data2/0_Arctus_Project/Ocean_Hackaton/data/synthetic_clouds"

    # Process all TIFF files
    process_all_tiffs(l1_toa_folder, output_folder)
