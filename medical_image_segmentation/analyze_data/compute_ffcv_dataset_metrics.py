import argparse

import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
import torch
from tqdm import tqdm


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
        pipelines={"image": [SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToTorchImage()], "label": [IntDecoder()]},
        distributed=False,
    )

    sum_ = 0
    sum_squared = 0
    num_pixels = 0

    # Iterate through the DataLoader
    for batch in tqdm(loader):
        images = batch[0].to(torch.float64)
        sum_ += torch.sum(images, dim=[0, 2, 3])  # Sum over batch, height, and width
        sum_squared += torch.sum(images**2, dim=[0, 2, 3])
        num_pixels += images.size(0) * images.size(2) * images.size(3)  # Batch size * Height * Width

    # Calculate mean and standard deviation
    mean = sum_ / num_pixels
    mean_of_squares = sum_squared / num_pixels
    mean_squared = mean**2
    std = torch.sqrt(mean_of_squares - mean_squared)

    return mean, std


def parse_args():
    """Creates args for command line interface"""
    parser = argparse.ArgumentParser(description="Shows images from .beton file.")
    parser.add_argument("--beton_path", type=str, help="Path to the .beton file to show images from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mean, std = compute_mean_and_std(args.beton_path)
    print(f"Mean: {mean}, Std: {std}")
