# from ffcv.fields import NDArrayField, FloatField
#
# class LinearRegressionDataset:
#     def __getitem__(self, idx):
#         return (X[idx], np.array(Y[idx]).astype('float32'))
#
#     def __len__(self):
#         return len(X)
#
# writer = DatasetWriter('/tmp/linreg_data.beton', {
#     'covariate': NDArrayField(shape=(D,), dtype=np.dtype('float32')),
#     'label': NDArrayField(shape=(1,), dtype=np.dtype('float32')),
# }, num_workers=16)
#
# writer.from_indexed_dataset(LinearRegressionDataset())
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
        image = image.resize(self.image_shape, Image.NEAREST)
        return np.asarray(image)

    def __len__(self):
        return len(self.image_paths)
