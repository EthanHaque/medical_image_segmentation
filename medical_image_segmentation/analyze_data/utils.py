from collections import Counter
from typing import List, Union, Callable
import os

import pydicom

from concurrent.futures import ProcessPoolExecutor, as_completed


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


def process_dicom_files(image_paths: List[str], processing_function: Callable[[str], dict], num_processes: int = 1) -> dict[str, dict]:
    """
    Processes DICOM files using the given processing function and returns the results as a dictionary where
    each key is a file path and each value is some information about the DICOM file.

    Parameters
    ----------
    image_paths : List[str] The paths of the DICOM files to process.
    processing_function : Callable[[str], dict] The processing function to apply to every DICOM file path.
    num_processes : int, optional [default = 1]: The number of processes to split the tasks among.

    Returns
    -------
    dict[str, dict]
        A dictionary where the key is the file path and the value is a dictionary with the results of
        the processing function.
    """
    if num_processes < 1:
        raise ValueError(f"num_threads must be greater than 1, but got {num_processes}")

    dicom_image_info = {}
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_file = {executor.submit(processing_function, file_path): file_path for file_path in image_paths}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error occurred during processing of {file_path}")
                raise
            dicom_image_info[file_path] = result

    return dicom_image_info


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

    dimension_information = process_dicom_files(image_paths, _get_dicom_image_dimensions_helper, num_processes)
    value_as_list = {}
    for key, value in dimension_information.items():
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
    return {
        "width": image_info.Rows,
        "height": image_info.Columns
    }


if __name__ == "__main__":
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    files = get_file_paths(dir_path, lambda path: path.endswith(".dcm"))[:10000]
    print(len(files))
    print(get_file_type_counts(dir_path))
