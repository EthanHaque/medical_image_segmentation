import argparse

import matplotlib.pyplot as plt
import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder
import numpy as np


def compute_mean_and_std(beton_file_path: str):
    """
    Reads images from a .beton file and computes the mean and standard deviation.

    Parameters
    ----------
    beton_file_path : str
        The path to the .beton file.
    """
    # Define the loader with the appropriate transforms
    order = ffcv.loader.OrderOption.SEQUENTIAL
    loader = Loader(
        beton_file_path,
        batch_size=128,
        num_workers=1,
        order=order,
        os_cache=True,
        drop_last=False,
        pipelines={"image": [SimpleRGBImageDecoder(), ffcv.transforms.ToTensor()]},
        distributed=False,
    )

    mean = 0
    nb_samples = 0
    for batch in loader:
        images = batch[0]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    print(f"Mean {mean:.4f}")


def parse_args():
    """Creates args for command line interface"""
    parser = argparse.ArgumentParser(description="Shows images from .beton file.")
    parser.add_argument("--beton_path", type=str, help="Path to the .beton file to show images from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_mean_and_std(args.beton_path)
