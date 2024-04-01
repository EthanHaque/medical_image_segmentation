import argparse

import matplotlib.pyplot as plt
from ffcv.loader import Loader
from ffcv.fields.decoders import SimpleRGBImageDecoder


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
    loader = Loader(
        beton_file_path,
        batch_size=1,
        num_workers=0,
        order="sequential",
        pipelines={"image": [SimpleRGBImageDecoder()]},
        distributed=False,
    )

    # Iterate over the loader and display the images
    for i, batch in enumerate(loader):
        if i >= num_images:
            break
        image = batch["image"].numpy().squeeze().transpose(1, 2, 0)
        plt.imshow(image)
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
