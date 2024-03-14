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


def write_subset(image_paths: List[str], map_output_path: str, image_output_directory: str,
                 dimensions_map: dict[str, List[int, int]], hashes_map: dict[str, str],
                 write_to_null: bool = False, num_subfolders: int = 0, num_processes: int = 1,
                 size: int = 1_000_000):
    possible_image_paths = pick_possible_images(image_paths, dimensions_map, hashes_map, min_size=256, max_size=768)
    dataset_map = map_paths_to_dataset(possible_image_paths)

    # These datasets are very large, so we only use a subset of them.
    not_whole = ["dukebreastcancer", "ctcolongraphy"]

    write_order = []
    for path, dataset in dataset_map.items():
        if dataset in not_whole:
            continue
        else:
            write_order.append(path)
    # randomizing these so the approximate remaining time is more accurate.
    random.shuffle(write_order)

    remaining_paths = []
    for path, dataset in dataset_map.items():
        if dataset in not_whole:
            remaining_paths.append(path)
    random.shuffle(remaining_paths)

    write_order = write_order.extend(remaining_paths)
    write_order = write_order[:size]

    statuses = utils.process_files(write_order, _write_subset_helper, num_processes, image_output_directory, write_to_null=write_to_null, num_subfolders=num_subfolders)

    success_count = 0
    original_path_to_new_path_map = {}
    for path, status in statuses.items():
        if status["error"]:
            continue
        else:
            success_count += 1
            original_path_to_new_path_map[status["image_path"]] = status["output_path"]

    with open(map_output_path, "w") as f:
        json.dump(original_path_to_new_path_map, f)

    print(f"Successfully wrote the pixel data from {success_count} DICOM images")


def _write_subset_helper(output_dir, image_path: str, write_to_null: bool = False,
                                  num_subfolders: int = 0) -> dict:
    """
    Helper function to write an individual DICOM image to output dir.

    Parameters
    ----------
    output_dir : str The root directory to write the images into.
    image_path: str The path to the DICOM image to write. The name of the writen file will
    be the hash of the DICOM image.
    write_to_null: bool [default: False] If true, writes the DICOM images to the null file.
    num_subfolders: int [default: 0] The number of folders to split the images into.

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
            if os.path.exists(output_path):
                raise ValueError(f"Path {output_path} already exists")
            image.save(output_path, format='PNG', bits=16)
        return {"image_path": image_path, "output_path": output_path, "error": None}
    except Exception as e:
        return {"image_path": image_path, "output_path": None, "error": e}


def pick_possible_images(image_paths: List[str], dimensions_map: dict[str, List[int, int]], hashes_map: dict[str, str],
                         min_size: int = 256, max_size: int = 768) -> List[str]:
    """
    Determines which file paths from the images could potentially be used to create a subset of data. Excludes
    duplicate images and images whose dimensions are too big or small.

    Parameters
    ----------
    image_paths : List[str] A list of file paths to DICOM images.
    dimensions_map : dict[str, List[int, int]] Maps file paths to their width and height.
    hashes_map : dict[str, str] Maps file paths to a unique hash.
    min_size : int, optional [default: 256] The smallest side length an image is allowed to have inclusive.
    max_size : int, optional [default: 768] The largest side length an image is allowed to have inclusive.

    Returns
    -------
    List[str]
        A list of file paths to DICOM images.
    """
    filtered_files = []
    taken_hashes = set()
    for path in image_paths:
        if (path not in dimensions_map) or (path not in hashes_map):
            continue

        width, height = dimensions_map[path]
        if (width > max_size) or (height > max_size) or (width < min_size) or (height < min_size):
            continue

        hash = hashes_map[path]
        if hash in taken_hashes:
            continue

        taken_hashes.add(hash)
        filtered_files.append(path)

    return filtered_files


def _get_dataset_name(image_path: str) -> str:
    """Gets the name of a dataset from the path to the dataset."""
    # This is highly specific to the naming convention. Need to change upon renaming folders.
    return image_path.split("med_datasets/")[1].split("/")[0]


def map_paths_to_dataset(image_paths: List[str]) -> dict[str, str]:
    """
    Maps image paths to the dataset they came from.

    Parameters
    ----------
    image_paths : List[str] A list of image paths

    Returns
    -------
    dict[str, str]
        A dictionary that maps image paths to the dataset they came from.
    """
    dataset_map = {}
    for path in image_paths:
        dataset_name = _get_dataset_name(path)
        dataset_map[dataset_name] = path

    return dataset_map


def get_dicom_image_dimensions_wrapper(dirs: List[str], output_path: str, num_processes: int = 1):
    """
    Wrapper to get the dimensions of all the pixel data from the dicom files in "dirs".

    Parameters
    ----------
    dirs : List[str] A list of directory paths to look for DICOM images.
    output_path : str A path where the hashes will be saved as JSON.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.
    """
    image_paths = utils.get_file_paths(dirs, lambda path: path.endswith(".dcm"))
    dimensions = get_dicom_image_dimensions(image_paths, num_processes)
    with open(output_path, "w") as f:
        json.dump(dimensions, f)

    count = 0
    for _, value in dimensions.items():
        if value:
            count += 1

    print(f"Successfully computed the dimensions of the pixel data from {count} DICOM images")


def get_dicom_image_dimensions(image_paths: List[str], num_processes: int = 1) -> dict[str, List[int]]:
    """
    Gets the width and height of every dicom file in the given list of image paths.

    Parameters
    ----------
    image_paths : List[str] A list of file paths to get the width and height of.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.

    Returns
    -------
    dict[str, List[int, int]]
        A dictionary where the keys are the file paths and the values are a list where the first
        element is the width and the second is the height of the DICOM image.
    """
    dimension_information = utils.process_files(image_paths, _get_dicom_image_dimensions_helper, num_processes)
    value_as_list = {}
    for key, value in dimension_information.items():
        if not value:
            continue
        value_as_list[key] = [value["width"], value["height"]]

    return value_as_list


def _get_dicom_image_dimensions_helper(image_path: str) -> dict:
    """
    Helper processing function to get DICOM image dimensions

    Parameters
    ----------
    image_path : str The path to the image.

    Returns
    -------
    dict
        A dictionary with two entries. One gives the image width and the other gives the image height.
    """
    image_info = pydicom.dcmread(image_path, stop_before_pixels=True)
    if hasattr(image_info, "Rows") and hasattr(image_info, "Columns"):
        return {"width": image_info.Rows, "height": image_info.Columns}
    else:
        return {}


def get_dicom_image_hashes(image_paths: List[str], num_processes: int = 1) -> dict[str, str]:
    """
    Gets the hashes of all the pixel data from the dicom files specified in `image_paths`.

    Parameters
    ----------
    image_paths : List[str] A list of file paths.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.

    Returns
    -------
        A dictionary where the keys are the file paths and the value is the hash of the pixel data.
    """
    dimension_information = utils.process_files(image_paths, _get_dicom_image_hashes_helper, num_processes)
    value_as_list = {}
    for key, value in dimension_information.items():
        if not value:
            continue
        value_as_list[key] = value["hash"]

    return value_as_list


def _get_dicom_image_hashes_helper(image_path: str) -> dict:
    """
    Helper processing function to get the hash of the imaging data from the DICOM file.

    Parameters
    ----------
    image_path : str The path to the image.

    Returns
    -------
    dict
        A dictionary with one entry, "hash".
    """
    try:
        arr = pydicom.dcmread(image_path).pixel_array
        arr.flags.writeable = False
        sha_hash = hashlib.sha256(arr).hexdigest()
    except Exception as e:
        return {}
    return {"hash": sha_hash}


def get_dicom_image_hashes_wrapper(dirs: List[str], output_path: str, num_processes: int = 1):
    """
    Wrapper to get the hashes of all the pixel data from the dicom files in "dirs".

    Parameters
    ----------
    dirs : List[str] A list of directory paths to look for DICOM images.
    output_path : str A path where the hashes will be saved as JSON.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.
    """
    image_paths = utils.get_file_paths(dirs, lambda path: path.endswith(".dcm"))
    hashes = get_dicom_image_hashes(image_paths, num_processes)
    with open(output_path, "w") as f:
        json.dump(hashes, f)

    count = 0
    for _, value in hashes.items():
        if value:
            count += 1

    print(f"Successfully computed the hashes of the pixel data from {count} DICOM images")


def parse_args():
    """Create args for command line interface."""
    parser = argparse.ArgumentParser(description="Process DICOM images and write them as raw images.")
    sub_parsers = parser.add_subparsers(help="Sub-commands", dest="subcommand")

    parser_get_dicom_hashes = sub_parsers.add_parser("dicom_hashes", help="Get the hashes of the dicom images.")
    parser_get_dicom_hashes.add_argument("--dirs", nargs="+", type=str, help="Directories to search DICOM images for.")
    parser_get_dicom_hashes.add_argument("--output_path", type=str, help="Where to write the map from paths to hashes")
    parser_get_dicom_hashes.add_argument("--num_processes", type=int,
                                         default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
                                         help="Number of processes to use for parallel processing.")

    parser_get_dicom_sizes = sub_parsers.add_parser("dicom_sizes", help="Get the sizes of the dicom images")
    parser_get_dicom_sizes.add_argument("--dirs", nargs="+", type=str, help="Directories to search DICOM images for.")
    parser_get_dicom_sizes.add_argument("--output_path", type=str, help="Where to write the map from paths to sizes")
    parser_get_dicom_sizes.add_argument("--num_processes", type=int,
                                        default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
                                        help="Number of processes to use for parallel processing.")

    parser_write_subset = sub_parsers.add_parser("write_subset", help="Write a subset of the DICOM images as PNGs.")
    parser_write_subset.add_argument("--dirs", nargs="+", type=str, help="Directories to search DICOM images for.")
    parser_write_subset.add_argument("--output_map_path", type=str, help="Where to write the map from the original DICOM images paths to the output image paths.")
    parser_write_subset.add_argument("--output_image_directory", type=str, help="Root directory to write the output images to.")
    parser_write_subset.add_argument("--size", type=int, help="Size of the subset to write.")
    parser_write_subset.add_argument("--num_subfolders", type=int, default=0, help="Number of subfolders to split the images into. If 0, does not write to any subfolders.")
    parser_write_subset.add_argument("--num_processes", type=int,
                                         default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
                                         help="Number of processes to use for parallel processing.")
    parser_write_subset.add_argument("--write_to_null", action="store_true", help="Write all images to null file.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.subcommand == "dicom_hashes":
        get_dicom_image_hashes_wrapper(args.dirs, args.output_path, num_processes=args.num_processes)
    if args.subcommand == "dicom_sizes":
        get_dicom_image_dimensions_wrapper(args.dirs, args.output_path, num_processes=args.num_processes)


if __name__ == "__main__":
    main()
