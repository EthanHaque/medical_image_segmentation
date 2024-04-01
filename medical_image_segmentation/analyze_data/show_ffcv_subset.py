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


if __name__ == "__main__":
    beton_file_path = "/scratch/gpfs/RUSTOW/med_datasets/ffcv_datasets/test_radiology_1M.beton"  # Update this to your .beton file path
    num_images = 5  # Number of images to display
    read_and_show_images(beton_file_path, num_images)
