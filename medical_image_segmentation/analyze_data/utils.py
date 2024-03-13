from collections import Counter
from functools import partial
from typing import List, Union, Callable, Tuple
import os

from PIL import Image

import pydicom

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


def get_file_paths(roots: Union[str, List[str]], matching_function: Callable[[str], bool]) -> list[Union[str, bytes]]:
    """
    Gathers all file paths from the given directories which end with the given filetypes.

    Parameters
    ----------
    roots : Union[str, List[str]] The root of the directories to recursively search.
    matching_function: Callable[[str], bool] Function that takes in a path and return true if it should be included
    in the output, false otherwise.

    Returns
    -------
    List[str]
        The absolute paths of the files in the given directories that end with one of the given filetypes.
    """
    if roots is None:
        return []
    if isinstance(roots, str):
        roots = [roots]

    absolute_file_paths = []
    for root in roots:
        for sub_root, dirs, files in os.walk(root):
            for file in files:
                path = os.path.join(sub_root, file)
                if matching_function(path):
                    absolute_file_paths.append(path)

    return absolute_file_paths


def get_file_type_counts(roots: Union[str, List[str]]) -> dict[str, int]:
    """
    Creates a frequency map of file extensions to their counts.

    Parameters
    ----------
    roots : Union[str, List[str]] The root of the directories to recursively search.

    Returns
    -------
    dict[str, int]
        A frequency map of file extensions to their counts.
    """
    files = get_file_paths(roots, lambda _: True)
    extensions = [os.path.splitext(file)[-1] for file in files]
    return dict(Counter(extensions))


def process_files(image_paths: List[str],
                  processing_function: Callable[[str, ...], dict],
                  num_processes: int = 1,
                  *args, **kwargs) -> dict[str, dict]:
    """
    Processes files using the given processing function and returns the results as a dictionary where
    each key is a file path and each value is some information about the file.

    Parameters
    ----------
    image_paths : List[str] The paths of the files to process.
    processing_function : Callable[[str], dict] The processing function to apply to every file path.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.

    Returns
    -------
    dict[str, dict]
        A dictionary where the key is the file path and the value is a dictionary with the results of
        the processing function.
    """
    if num_processes < 1:
        raise ValueError(f"num_processes must be greater than 1, but got {num_processes}")

    image_info = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        partial_processing_function = partial(processing_function, *args, **kwargs)
        future_to_file = {}

        with Progress(
                TextColumn("[bold blue]{task.completed}/{task.total} files batched"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Batching files...", total=len(image_paths), file_count=0)
            for i, file_path in enumerate(image_paths):
                future_to_file[executor.submit(partial_processing_function, file_path)] = file_path
                progress.update(task, advance=1)
                

        with Progress(
                TextColumn("[bold blue]{task.completed}/{task.total} files processed"),
                BarColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Processing files...", total=len(image_paths), file_count=0)
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                except Exception:
                    raise RuntimeError(f"Error occurred during processing of {file_path}")
                image_info[file_path] = result
                progress.update(task, advance=1)

    return image_info


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
    dimension_information = process_files(image_paths, _get_dicom_image_dimensions_helper, num_processes)
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


def get_raster_image_dimensions(image_paths: List[str], num_processes: int = 1) -> dict[str, List[int]]:
    """
    Gets the width and height of every image file in the given list of image paths.

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
    dimension_information = process_files(image_paths, _get_raster_image_dimensions_helper, num_processes)
    value_as_list = {}
    for key, value in dimension_information.items():
        if not value:
            continue
        value_as_list[key] = [value["width"], value["height"]]

    return value_as_list


def _get_raster_image_dimensions_helper(image_path: str) -> dict:
    """
    Helper processing function to get raster image dimensions.

    Parameters
    ----------
    image_path : str The path to the image.

    Returns
    -------
    dict
        A dictionary with two entries. One gives the image width and the other gives the image height.
    """
    im = Image.open(image_path)
    width, height = im.size
    return {"width": width, "height": height}


if __name__ == "__main__":
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    files = get_file_paths(dir_path, lambda path: path.endswith(".dcm"))[:10000]
    print(len(files))
    print(get_file_type_counts(dir_path))
