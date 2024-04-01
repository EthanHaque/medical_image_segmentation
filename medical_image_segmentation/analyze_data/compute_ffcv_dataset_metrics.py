import argparse

import matplotlib.pyplot as plt
import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder
import numpy as np
import torch


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
        pipelines={"image": [SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToTorchImage()]},
        distributed=False,
    )

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:
        data = data[0]
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def parse_args():
    """Creates args for command line interface"""
    parser = argparse.ArgumentParser(description="Shows images from .beton file.")
    parser.add_argument("--beton_path", type=str, help="Path to the .beton file to show images from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mean, std = compute_mean_and_std(args.beton_path)
    print(f"Mean:{mean}, Std:{std}")
