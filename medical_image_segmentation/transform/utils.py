from typing import List, Union, Any, Tuple
import os

import matplotlib.pyplot as plt
from pydicom import dcmread


def gather_files(roots: Union[str, List[str]], filetypes: Union[str, Tuple[str]]) -> Union[list[Any], list[Union[str, bytes]]]:
    """
    Gathers all files from the given directories which end with the given filetypes.

    Parameters
    ----------
    roots : Union[str, List[str]] or str The root of the directories to recursively search.
    filetypes : Union[str, Tuple[str]] or str The filetypes to include in the output

    Returns
    -------
    List[str]
        The absolute paths of the files in the given directories that end with one of the given filetypes.
    """
    if filetypes is None or roots is None:
        return []
    if isinstance(filetypes, str) or isinstance(filetypes, list):
        filetypes = tuple(filetypes)
    if isinstance(roots, str):
        roots = [roots]

    absolute_file_paths = []
    for root in roots:
        for sub_root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(filetypes):
                    absolute_file_paths.append(os.path.join(sub_root, file))

    return absolute_file_paths



if __name__ == '__main__':
    # path = "/scratch/gpfs/eh0560/data/med_datasets/cptacccrcc/manifest-1692379830142/CPTAC-CCRCC/C3N-03019/11-01-2000-NA-KT miednicy mniejszej - bad z kontrastem-00143/10.000000-ZYLNA  5.0  I30f  3-89793/1-091.dcm"
    # ds = dcmread(path)
    # arr = ds.pixel_array
    #
    # plt.imshow(arr, cmap="gray")
    # plt.show()
    dir_path = ["/scratch/gpfs/eh0560/data/med_datasets/cptacccrcc/manifest-1692379830142/CPTAC-CCRCC/C3N-03019"]
    exts = [".dcm"]
    print(gather_files(dir_path, exts))
