from typing import Tuple, List
import os
import argparse
import nibabel as nib
import numpy as np
import cv2
from rich.progress import Progress
from concurrent.futures import ProcessPoolExecutor, as_completed

from medical_image_segmentation.analyze_data.utils import get_file_paths

class NIBSegmentationFile():
    def __init__(self, nib_file_path, is_mask):
        self.file_path = nib_file_path
        self.is_mask = is_mask
        self.image = nib.load(nib_file_path)

    def get_shape(self):
        return self.image.shape

    def get_arr(self):
        return self.image.get_fdata()

def get_scan_and_mask_pairs(scan_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """Pairs the scan and mask paths into a list of tuples (scan, mask)."""
    image_files = get_file_paths(scan_dir, lambda x: x.endswith(".nii.gz"))
    mask_files = get_file_paths(mask_dir, lambda x: x.endswith(".nii.gz"))

    mask_uid_to_path = {path.split("/")[-1]: path for path in mask_files}

    image_mask_paris = []
    for image_file_path in image_files:
        uid = image_file_path.split("/")[-1]
        mask_file_path = mask_uid_to_path[uid]
        image_mask_paris.append((image_file_path, mask_file_path))

    return image_mask_paris


def load_file_headers(paths):
    return [nib.load(path) for path in paths]


def get_slice_output_path(segmentation_file: NIBSegmentationFile, output_dir: str, slice_number: int) -> str:
    """Gets output path for a single slice of a nifti file."""
    file_name = segmentation_file.file_path.split("/")[-1]
    uid_no_ext = file_name.split(".")[0]
    uid_no_ext = uid_no_ext + f"_{slice_number}"
    output_file_name = uid_no_ext + ".png"
    return os.path.join(output_dir, output_file_name)


def save_nii_slices(segmentation_file: NIBSegmentationFile, output_dir: str, slice_dim: int):
    """Saves slices of nifti file to an output directory."""
    image_arr = segmentation_file.get_arr()
    num_slices = image_arr.shape[slice_dim]

    for slice_number in range(num_slices):
        slice = image_arr.take(indices=slice_number, axis=slice_dim)
        # if not is_mask:
        #     slice = ((slice - slice.min()) / (slice.max() - slice.min()) * 255)
        # else:
        #     slice = (slice != 0).astype(np.uint8) * 255
        # slice = slice.astype(np.uint8)
        # output_path = get_slice_output_path(segmentation_file.file_path, output_dir, slice_number)
        # cv2.imwrite(output_path, slice)


def main(scan_dir: str, mask_dir: str, root_output_dir: str, slice_dim: int, max_workers: int):
    pairs = get_scan_and_mask_pairs(scan_dir, mask_dir)
    files = []
    for image_path, mask_path in pairs:
        files.append(NIBSegmentationFile(image_path, False))
        files.append(NIBSegmentationFile(mask_path, True))

    image_output_dir = os.path.join(root_output_dir, "images")
    masks_output_dir = os.path.join(root_output_dir, "masks")

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers) as executor:
        futures = []
        for file in files:
            output_dir = masks_output_dir if file.is_mask else image_output_dir
            futures.append(executor.submit(save_nii_slices, file, output_dir, slice_dim))
        with Progress() as progress:
            main_task_id = progress.add_task("[cyan]Processing images and masks...", total=len(futures))
            for _ in as_completed(futures):
                progress.update(main_task_id, advance=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, help="Root directory with nifti files of scans.")
    parser.add_argument("--mask_dir", type=str, help="Root directory with nifti files of masks.")
    parser.add_argument("--root_output_dir", type=str, help="Where to write images and masks to.")
    parser.add_argument("--slice_dim", type=int, default=2, help="Which dimension to slice along.")
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Max number of workers to use for parallel processing."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.scan_dir, args.mask_dir, args.root_output_dir, args.slice_dim, args.max_workers)
