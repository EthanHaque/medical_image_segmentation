import argparse

import math
import matplotlib.pyplot as plt
import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
from ffcv.transforms import ToTensor, ToTorchImage
import numpy as np
import torchvision.utils as vutils


def read_and_show_images(beton_file_path: str, num_images: int):
    """
    Reads images from a .beton file and displays them.

    Parameters
    ----------
    beton_file_path : str
        The path to the .beton file.
    num_images : int
        The number of images to display.
    """
    # Define the loader with the appropriate transforms
    order = ffcv.loader.OrderOption.SEQUENTIAL
    loader = Loader(
        beton_file_path,
        batch_size=num_images,
        num_workers=1,
        order=order,
        os_cache=True,
        drop_last=False,
        pipelines={"image": [SimpleRGBImageDecoder(), ToTensor(), ToTorchImage()], "label": [IntDecoder()]},
        distributed=False,
    )
    grid_size = math.ceil(math.sqrt(num_images))
    images, labels = next(iter(loader))
    grid = vutils.make_grid(images)
    grid_np = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.savefig("/bind_tmp/image_grid.png")
    plt.close()


def parse_args():
    """Creates args for command line interface"""
    parser = argparse.ArgumentParser(description="Shows images from .beton file.")
    parser.add_argument("--beton_path", type=str, help="Path to the .beton file to show images from.")
    parser.add_argument("--num_images", type=int, help="Number of images to show.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    read_and_show_images(args.beton_path, args.num_images)
