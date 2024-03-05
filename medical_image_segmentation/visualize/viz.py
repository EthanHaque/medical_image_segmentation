import medical_image_segmentation.analyze_data.utils as utils
import matplotlib.pyplot as plt

import numpy as np

dimensions = get_dicom_image_dimensions(files, num_threads=4)
    widths = []
    heights = []
    for file, dims in dimensions.items():
        widths.append(dims[0])
        heights.append(dims[1])

    widths = np.array(widths)
    heights = np.array(heights)

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, c=np.log(widths * heights), cmap="cool", alpha=0.9)
    plt.colorbar(label="Log of Area (pixels^2)")
    plt.title("Distribution of DICOM Image Dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.tight_layout()

    # Show the plot
    plt.show()
