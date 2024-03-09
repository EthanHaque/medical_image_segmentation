import hashlib
import json
import os
import random
from typing import List
import argparse

from PIL import Image
import numpy as np
import pydicom

import medical_image_segmentation.analyze_data.utils as utils


def get_subset_dicom_image_paths(size: int) -> List[str]:
    """
    Get a list of paths to the dicom images to use in this project.

    Parameters
    ----------
    size : int
        Size of the subset to be returned.

    Returns
    -------
    List[str]
        A list of file paths.
    """
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
    upper_bound = 768
    lower_bound = 256
    for path, dim in dimensions.items():
        if lower_bound <= dim[0] <= upper_bound and lower_bound <= dim[1] <= upper_bound:
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

    remaining = size - len(subset_file_paths)

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

    return sorted(subset_file_paths)


def write_raw_image_subset(image_paths: List[str], output_dir: str, num_processes: int = 1, **kwargs) -> int:
    """
    Reads DICOM images from the image_paths and writes them as raw images to output_dir.

    Parameters
    ----------
    output_dir : str The path to the output directory to save the raw images.
    image_paths : List[str] A list of file paths to DICOM images.
    num_processes : int The number of cores to batch tasks on.

    Returns
    -------
    int
        The number of images successfully written.
    """
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate paths contained in image_paths.")
    os.makedirs(output_dir, exist_ok=True)

    statuses = utils.process_dicom_files(image_paths, write_raw_image_subset_helper, num_processes, output_dir, **kwargs)

    count = 0
    for path, status in statuses.items():
        if status["error"] is None:
            count += 1
        else:
            print(status)

    return count


def write_raw_image_subset_helper(output_dir, image_path: str, write_to_null: bool = False, **kwargs) -> dict:
    """
    Helper function to write an individual DICOM image to output dir.

    Parameters
    ----------
    image_path: str The path to the DICOM image to write. The name of the writen file will
    be the hash of the DICOM image.
    write_to_null: bool [default: False] If true, writes the DICOM images to the null file.

    Returns
    -------
    dict
        A dictionary with three keys:
        - image_path The path to the source DICOM image.
        - output_path The path to the destination image. Is None if the image is invalid.
        - error An error message if the image is invalid. Is None if no errors occurred.
    """
    arr = pydicom.dcmread(image_path).pixel_array
    arr.flags.writeable = False
    sha_hash = hashlib.sha256(arr).hexdigest()
    output_path = os.path.join(output_dir, f"{sha_hash}.png")

    try:
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        if max_val > min_val:
            arr = ((arr - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Invalid image data for image {image_path}")
    except ValueError as e:
        return {"image_path": image_path, "output_path": None, "error": e}

    try:
        image = Image.fromarray(arr)
        if write_to_null:
            with open("/dev/null", "wb") as f:
                image.save(f, format='PNG', bits=16)
        else:
            image.save(output_path, format='PNG', bits=16)
        return {"image_path": image_path, "output_path": output_path, "error": None}
    except Exception as e:
        return {"image_path": image_path, "output_path": output_path, "error": e}


def create_subset(size: int, output_path: str):
    """
    Create a subset of the images.

    Parameters
    ----------
    size : int The size of the subset.
    output_path : str The path to write the subset to.
    """
    files = get_subset_dicom_image_paths(size)
    with open(output_path, "w") as f:
        for image_path in files:
            f.write(image_path + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Process DICOM images and write them as raw images.")
    parser.add_argument("--num_processes", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
                        help="Number of processes to use for parallel processing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    subset_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/image_path_list"
    with open(subset_path, "r") as f:
        paths = f.read().splitlines()
        paths = [path.strip() for path in paths]

    write_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/test"

    # Randomizing to make expected remaining time more accurate.
    random.shuffle(paths)
    count = write_raw_image_subset(paths[:100], write_path, num_processes=args.num_processes, write_to_null=True)
    print(count)
