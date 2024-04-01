import argparse

import matplotlib.pyplot as plt
import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder
import numpy as np


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
        batch_size=1,
        num_workers=1,
        order=order,
        os_cache=True,
        drop_last=False,
        pipelines={"image": [SimpleRGBImageDecoder()]},
        distributed=False,
    )

    # Iterate over the loader and display the images
    images = []
    for i, batch in enumerate(loader):
        if i >= num_images:
            break
        image = batch[0]
        image = np.squeeze(image)
        images.append(image)

    for i, im in enumerate(images):
        plt.imshow(im)
        plt.title(f"Image {i + 1}")
        plt.axis("off")
        plt.show()


def parse_args():
    """Creates args for command line interface"""
    parser = argparse.ArgumentParser(description="Shows images from .beton file.")
    parser.add_argument("--beton_path", type=str, help="Path to the .beton file to show images from.")
    parser.add_argument("--num_images", type=int, help="Number of images to show.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    read_and_show_images(args.beton_path, args.num_images)
