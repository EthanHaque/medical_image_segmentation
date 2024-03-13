import multiprocessing
import threading
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
    with ProcessPoolExecutor(max_workers=num_processes, initializer=start_orphan_checker) as executor:
        partial_processing_function = partial(processing_function, *args, **kwargs)
        future_to_file = {}

        with Progress(
            TextColumn("[bold blue]{task.completed}/{task.total} files batched"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
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
                except Exception as e:
                    for f in future_to_file.keys():
                        f.cancel()
                    raise RuntimeError(f"Error occurred during processing of {file_path}: {e}")

                image_info[file_path] = result
                progress.update(task, advance=1)

    return image_info


def start_orphan_checker():
    """Checks for orphaned child processes and kills them."""
    def exit_if_orphaned():
        multiprocessing.parent_process().join()
        os._exit(-1)

    threading.Thread(target=exit_if_orphaned, daemon=True).start()



if __name__ == "__main__":
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets", "/scratch/gpfs/RUSTOW/med_datasets"]
    files = get_file_paths(dir_path, lambda path: path.endswith(".dcm"))[:10000]
    print(len(files))
    print(get_file_type_counts(dir_path))
