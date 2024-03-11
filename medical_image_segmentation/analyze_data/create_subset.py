import hashlib
import json
import os
import random
import warnings
from typing import List, Tuple
import argparse

from PIL import Image
import numpy as np
import pydicom

import medical_image_segmentation.analyze_data.utils as utils


def get_subset_dicom_image_paths(size: int, num_processes: int = 1) -> List[str]:
    """
    Get a list of paths to the dicom images to use in this project.

    Parameters
    ----------
    size : int Size of the subset to be returned.
    num_processes : int The number of cores to batch tasks on.

    Returns
    -------
    List[str]
        A list of file paths.
    """
    datasets_root_paths = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    image_dimensions_json_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/dicom_image_dimensions.json"

    # Get image dimensions file if it exists, otherwise create it.
    if os.path.isfile(image_dimensions_json_path):
        with open(image_dimensions_json_path, "r") as f:
            dimensions = json.load(f)
    else:
        files = utils.get_file_paths(datasets_root_paths, lambda path: path.endswith(".dcm"))
        random.shuffle(files)
        dimensions = utils.get_dicom_image_dimensions(files, num_processes=num_processes)
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


def write_raw_image_subset(image_paths: List[str], output_dir: str, num_processes: int = 1, **kwargs) -> Tuple[int, dict]:
    """
    Reads DICOM images from the image_paths and writes them as raw images to output_dir.

    Parameters
    ----------
    output_dir : str The path to the output directory to save the raw images.
    image_paths : List[str] A list of file paths to DICOM images.
    num_processes : int The number of cores to batch tasks on.

    Returns
    -------
    Tuple[int, dict]
        The number of images successfully processed, and a dictionary mapping the input paths to the output paths.
    """
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Duplicate paths contained in image_paths.")
    os.makedirs(output_dir, exist_ok=True)

    statuses = utils.process_dicom_files(image_paths, write_raw_image_subset_helper, num_processes, output_dir, **kwargs)

    count = 0
    image_path_map = {}
    for path, status in statuses.items():
        if status["error"] is None:
            count += 1
            image_path_map[status["image_path"]] = status["output_path"]
        else:
            print(status)

    return count, image_path_map


def write_raw_image_subset_helper(output_dir, image_path: str, write_to_null: bool = False, num_subfolders: int = 0) -> dict:
    """
    Helper function to write an individual DICOM image to output dir.

    Parameters
    ----------
    output_dir : str The root directory to write the images into.
    image_path: str The path to the DICOM image to write. The name of the writen file will
    be the hash of the DICOM image.
    write_to_null: bool [default: False] If true, writes the DICOM images to the null file.
    num_subfolders: int [default: 1] The number of folders to split the images into.

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
    os.makedirs(output_dir, exist_ok=True)

    if num_subfolders > 0 and not write_to_null:
        subfolder_index = int(sha_hash, 16) % num_subfolders
        subfolder_name = f"{subfolder_index:0{len(str(num_subfolders - 1))}}"
        subfolder_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        output_path = os.path.join(subfolder_path, f"{sha_hash}.png")
    else:
        output_path = os.path.join(output_dir, f"{sha_hash}.png")

    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    try:
        if max_val > min_val:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    arr = ((arr - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
                except RuntimeWarning as e:
                    return {"image_path": image_path, "output_path": None, "error": e}
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


def create_subset(size: int, output_path: str, num_processes: int = 1):
    """
    Create a subset of the images.

    Parameters
    ----------
    size : int The size of the subset.
    output_path : str The path to write the subset to.
    num_processes : int  The number of processes to use.
    """
    files = get_subset_dicom_image_paths(size, num_processes=num_processes)
    with open(output_path, "w") as f:
        for image_path in files:
            f.write(image_path + "\n")


def finalize_image_subset(size: int, original_image_to_new_image_map: dict, output_path: str):
    """
    Given a map of original image paths to their new written image paths, determines which images to include in the
    finalized subset of the images.

    Parameters
    ----------
    size : int The size of the subset to create.
    original_image_to_new_image_map : dict The original image paths mapped to their new image paths.
    output_path : str The path to write the output file containing a map between the new image paths and the original image paths.
    """
    if len(original_image_to_new_image_map) < size:
        raise ValueError(f"Not enough images to create subset of specified size. Has only {len(original_image_to_new_image_map)} images,"
                         f"but requires {size}")

    dataset_file_map = {}
    for image_path in original_image_to_new_image_map:
        suffix = image_path.split("med_datasets/")[1]
        dataset_name = suffix.split("/")[0]
        if dataset_name not in dataset_file_map:
            dataset_file_map[dataset_name] = []
        dataset_file_map[dataset_name].append(image_path)

    remaining_size = size
    for dataset_files in dataset_file_map.values():
        if dataset_files not in ["dukebreastcancer", "ctcolongraphy"]:
            remaining_size -= len(dataset_files)

    duke_size = remaining_size // 2
    colon_size = remaining_size - duke_size

    duke_files = dataset_file_map["dukebreastcancer"]
    random.shuffle(duke_files)
    duke_files = duke_files[:duke_size]

    colon_files = dataset_file_map["ctcolongraphy"]
    random.shuffle(colon_files)
    colon_files = colon_files[:colon_size]

    final_images_source = []
    final_images_source.extend(colon_files)
    final_images_source.extend(duke_files)
    for dataset_name in dataset_file_map:
        if dataset_name not in ["dukebreastcancer", "ctcolongraphy"]:
            final_images_source.extend(dataset_file_map[dataset_name])

    final_images_destination = []
    for original_image in final_images_source:
        final_images_destination.append(original_image_to_new_image_map[original_image])

    with open(output_path, "w") as f:
        for image_path in final_images_destination:
            f.write(image_path + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Process DICOM images and write them as raw images.")
    parser.add_argument("--num_processes", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
                        help="Number of processes to use for parallel processing.")
    parser.add_argument("--subset_size", type=int, default=1_050_000, help="Size of the subset to write")
    parser.add_argument("--create_subset", action="store_true", help="Flag to enable subset creation")
    parser.add_argument("--write_to_null", action="store_true", help="Flag to disable writing images to output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    subset_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/possible_image_paths"
    if args.create_subset:
        create_subset(args.subset_size, subset_path, args.num_processes)

    with open(subset_path, "r") as f:
        paths = f.read().splitlines()
        paths = [path.strip() for path in paths]

    write_path = "/scratch/gpfs/eh0560/data/med_datasets/segmentation_png"

    # Randomizing to make expected remaining time more accurate.
    random.shuffle(paths)
    count, input_output_path_map = write_raw_image_subset(paths, write_path, num_processes=args.num_processes, write_to_null=args.write_to_null, num_subfolders=100)

    input_output_path_map_json_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/input_output_path_map.json"
    with open(input_output_path_map_json_path, "w") as f:
        json.dump(input_output_path_map, f)

    print(count)

    final_subset_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/final_image_paths"
    finalize_image_subset(1_000_000, input_output_path_map, final_subset_path)

if __name__ == "__main__":
    # main()
    with open("/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/input_output_path_map.json", "r") as f:
        input_output_path_map = json.load(f)

    final_subset_path = "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/dicom_image_analysis_info/final_image_paths"
    finalize_image_subset(1_000_000, input_output_path_map, final_subset_path)
