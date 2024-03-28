import os

from ffcv.fields import NDArrayField
from ffcv.writer import DatasetWriter

import argparse
import json
from typing import Tuple, List

import numpy as np
import pydicom
from PIL import Image


def get_image_paths(original_to_new_map_path: str) -> List[str]:
    """
    Gets the DICOM image paths from the map of the original file paths to
    new file paths.

    Parameters
    ----------
    original_to_new_map_path : str The path to a json file that maps the original
    DICOM image path to the new image path.

    Returns
    -------
    List[str]
        A list of DICOM image paths.
    """
    with open(original_to_new_map_path, "r") as f:
        original_to_new_map = json.load(f)

    return list(original_to_new_map.keys())


class DICOMImageDataset:
    def __init__(self, image_paths: List[str], output_shape: Tuple[int, int]):
        self.image_paths = image_paths
        self.image_shape = output_shape

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_arr = pydicom.dcmread(image_path).pixel_array
        arr_min = image_arr.min()
        arr_max = image_arr.max()
        image_arr = ((image_arr - arr_min) / (arr_max - arr_min)).astype(np.float32)

        image = Image.fromarray(image_arr)
        image = image.resize(self.image_shape, Image.BICUBIC)

        # normalizing again after resizing
        image_arr = np.asarray(image)
        arr_min = image_arr.min()
        arr_max = image_arr.max()
        image_arr = ((image_arr - arr_min) / (arr_max - arr_min)).astype(np.float32)

        # wrapping in tuple so that return value has correct shape (size 1).
        return (image_arr,)

    def __len__(self):
        return len(self.image_paths)


def parse_args():
    """Create args for command line interface."""
    parser = argparse.ArgumentParser(description="Process DICOM images and write them as a ffcv dataset.")
    parser.add_argument(
        "--original_to_new_map_path",
        type=str,
        help="Map from original image paths to new image paths. Used to find the correct DICOM images to use.",
    )
    parser.add_argument("--output_file_path", type=str, help="Path to write .beton file to.")
    parser.add_argument("--width", type=int, help="Width to resize the images to.")
    parser.add_argument("--height", type=int, help="Height to resize the images to.")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")),
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument("--test", action="store_true", help="Makes a small subset.")

    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = get_image_paths(args.original_to_new_map_path)
    if args.test:
        image_paths = image_paths[:100]

    resize_dimensions = (args.height, args.width)

    writer = DatasetWriter(
        args.output_file_path,
        {"image": NDArrayField(shape=resize_dimensions, dtype=np.dtype("float32"))},
        num_workers=args.num_processes,
    )

    dataset = DICOMImageDataset(image_paths, resize_dimensions)

    writer.from_indexed_dataset(dataset)


if __name__ == "__main__":
    main()
