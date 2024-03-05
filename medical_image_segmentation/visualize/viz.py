from collections import Counter
from typing import List

import medical_image_segmentation.analyze_data.utils as utils
import matplotlib.pyplot as plt
import numpy as np


def plot_image_shapes(image_shapes: List[List[int]]):
    """
    Plot image sizes on a scatter plot.

    Parameters
    ----------
    image_shapes : List[int] List of width, height pairs.
    """
    widths = []
    heights = []
    for dims in image_shapes:
        widths.append(dims[0])
        heights.append(dims[1])

    widths = np.array(widths)
    heights = np.array(heights)

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, c=np.log(widths * heights), cmap="cool")
    plt.colorbar(label="Log of Area (pixels^2)")
    plt.title("Distribution of DICOM Image Dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.tight_layout()

    plt.show()


def plot_image_sizes(image_shapes: List[List[int]]):
    """
    Plot a bar chart of image sizes.

    Parameters
    ----------
    image_shapes : List[int] List of width, height pairs.
    """
    sizes = [dims[0] for dims in image_shapes]
    size_counts = Counter(sizes)

    # Sort the sizes and counts for a nicer plot
    sorted_sizes = sorted(size_counts.keys())
    sorted_counts = [size_counts[size] for size in sorted_sizes]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_sizes, sorted_counts, color="darkorchid")
    plt.title("Distribution of DICOM Image Sizes")
    plt.xlabel("Size (pixels)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    files = utils.get_file_paths(dir_path, lambda path: path.endswith(".dcm"))[:10000]
    dimensions = utils.get_dicom_image_dimensions(files, num_threads=4)
    plot_image_shapes(list(dimensions.values()))
    plot_image_sizes(list(dimensions.values()))
