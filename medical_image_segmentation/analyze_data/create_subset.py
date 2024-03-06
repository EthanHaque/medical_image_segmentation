import json
import os
import random
from typing import List

import medical_image_segmentation.analyze_data.utils as utils


def get_subset_dicom_image_paths() -> List[str]:
    random.seed(2)

    datasets_root_paths = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    image_dimensions_json_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/dicom_image_dimensions.json"

    # Get image dimensions file if it exists, otherwise create it.
    if os.path.isfile(image_dimensions_json_path):
        with open(image_dimensions_json_path, "r") as f:
            dimensions = json.load(f)
    else:
        files = utils.get_file_paths(datasets_root_paths, lambda path: path.endswith(".dcm"))
        dimensions = utils.get_dicom_image_dimensions(files, num_processes=8)
        with open(image_dimensions_json_path, "w") as f:
            json.dump(dimensions, f)

    dimensions_without_outliers = {}
    threshold = 768
    for path, dim in dimensions.items():
        if dim[0] <= threshold and dim[1] <= threshold:
            dimensions_without_outliers[path] = dim

    # Map the datasets to image paths.
    dataset_names = ["dukebreastcancer", "remind", "tcgakirc", "midrc", "cptacccrcc", "hcctace", "cctumor", "tcgablca",
                     "pancreasct", "ctcolongraphy"]
    dataset_file_map = {dataset_name: [] for dataset_name in dataset_names}
    for image_path in dimensions_without_outliers:
        suffix = image_path.split("med_datasets/")[1]
        dataset_name = suffix.split("/")[0]
        dataset_file_map[dataset_name].append(image_path)

    subset_file_paths = []
    for dataset_name in [x for x in dataset_names if x not in ["dukebreastcancer", "ctcolongraphy"]]:
        subset_file_paths.extend(dataset_file_map[dataset_name])

    target_size = 1_000_000
    remaining = target_size - len(subset_file_paths)

    duke_size = remaining // 2
    colongraphy_size = remaining - duke_size

    duke_list = dataset_file_map["dukebreastcancer"]
    random.shuffle(duke_list)
    duke_subset = duke_list[:duke_size]
    subset_file_paths.extend(duke_subset)

    colongraphy_list = dataset_file_map["ctcolongraphy"]
    random.shuffle(colongraphy_list)
    colongraphy_subset = colongraphy_list[:colongraphy_size]
    subset_file_paths.extend(colongraphy_subset)

    return subset_file_paths


if __name__ == "__main__":
    print(len(get_subset_dicom_image_paths()))
