from typing import List

import medical_image_segmentation.analyze_data.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_image_shapes(image_shapes: List[List[int]]):
    """
    Plot image sizes on a scatter plot, with more frequent sizes represented by larger points.

    Parameters
    ----------
    image_shapes : List[List[int]] List of width, height pairs.
    """
    shape_counter = Counter(tuple(shape) for shape in image_shapes)
    widths, heights, counts = zip(*[(shape[0], shape[1], count) for shape, count in shape_counter.items()])

    widths = np.array(widths)
    heights = np.array(heights)
    counts = np.array(counts)

    scale = 10_000
    point_sizes = np.clip(counts / np.max(counts) * scale, 1, scale)

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, s=point_sizes, c=np.sqrt(widths * heights), cmap="cool", alpha=0.8)
    plt.colorbar(label="Sqrt of Area (pixels^2)")
    plt.title("Distribution of DICOM Image Dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    files = utils.get_file_paths(dir_path, lambda path: path.endswith(".dcm"))[1_000_000:1_500_000]
    dimensions = utils.get_dicom_image_dimensions(files, num_processes=8)
    plot_image_shapes(list(dimensions.values()))