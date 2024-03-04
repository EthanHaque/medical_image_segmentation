from collections import Counter
from typing import List, Union, Any, Tuple, Callable
import os

import matplotlib.pyplot as plt
from pydicom import dcmread


def gather_files(roots: Union[str, List[str]], matching_function: Callable[[str], bool]) -> list[Union[str, bytes]]:
    """
    Gathers all files from the given directories which end with the given filetypes.

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
    files = gather_files(roots, lambda _: True)
    extensions = [os.path.splitext(file)[-1] for file in files]
    return dict(Counter(extensions))


if __name__ == '__main__':
    # path = "/scratch/gpfs/eh0560/data/med_datasets/cptacccrcc/manifest-1692379830142/CPTAC-CCRCC/C3N-03019/11-01-2000-NA-KT miednicy mniejszej - bad z kontrastem-00143/10.000000-ZYLNA  5.0  I30f  3-89793/1-091.dcm"
    # ds = dcmread(path)
    # arr = ds.pixel_array
    #
    # plt.imshow(arr, cmap="gray")
    # plt.show()
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets/cptacccrcc/manifest-1692379830142/CPTAC-CCRCC/C3N-03019"]
    print(get_file_type_counts(dir_path))
