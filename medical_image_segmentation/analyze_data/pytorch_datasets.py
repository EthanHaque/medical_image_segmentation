import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os
from medical_image_segmentation.analyze_data.utils import get_file_paths
import numpy as np


class ChestXRayDataset(Dataset):
    """
    A PyTorch dataset for chest X-ray images.

    Attributes
    ----------
    data_frame : pd.DataFrame
        A DataFrame containing the labels and image paths.
    transform : torchvision.transforms.Compose
        A composition of transformations to apply to the images.
    label_encoding : Dict[str, int]
        A dictionary mapping labels to integers.

    Methods
    -------
    __len__() -> int
        Returns the number of samples in the dataset.
    __getitem__(index: int) -> Tuple[torch.Tensor, int]
        Returns the image and its label (as an integer) at the specified index.
    """

    def __init__(self, csv_file: str, transform=None):
        """
        Parameters
        ----------
        csv_file : str
            The path to the CSV file containing the labels and image paths.
        transform : torchvision.transforms.Compose, optional
            A composition of transformations to apply to the images (default is None).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.label_encoding = self._create_label_encoding()

    def _create_label_encoding(self) -> Dict[str, int]:
        """Creates a dictionary mapping labels to integers."""
        labels = self.data_frame["label"].unique()
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the image and its label (as an integer) at the specified index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple containing the image as a torch.Tensor and its label as an integer.
        """
        img_path = self.data_frame.iloc[index, 1]
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame.iloc[index, 0]
        label_encoded = self.label_encoding[label]

        if self.transform:
            image = self.transform(image)

        return image, label_encoded


class Radiology1MDataset(Dataset):
    """
    A PyTorch dataset for radiological images.

    Attributes
    ----------
    data_root: str
        Root directory where all .png images are located.
    file_paths: List[str]
        List of file paths to all .png images in the dataset.
    transform : torchvision.transforms.Compose
        A composition of transformations to apply to the images.

    Methods
    -------
    __len__() -> int
        Returns the number of samples in the dataset.
    __getitem__(index: int) -> Tuple[torch.Tensor]
        Returns the image inside a tuple.
    """

    def __init__(self, data_root: str, transform=None):
        """
        Parameters
        ----------
        data_root : str
            Path where all .png images are located.
        transform : torchvision.transforms.Compose, optional
            A composition of transformations to apply to the images (default is None).
        """
        self.data_root = data_root
        self.file_paths = get_file_paths(data_root, lambda x: x.endswith(".png"))
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor,]:
        """
        Returns the image and its label (as an integer) at the specified index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, int]
            A tuple containing the image as a torch.Tensor and its label as an integer.
        """
        img_path = self.file_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return (image,)


class DecathlonDataset(Dataset):
    """
    A PyTorch dataset for image segmentation using the Medical Decathlon dataset.

    Attributes
    ----------
    image_paths: List[str]
        List of paths to each image.
    mask_paths: List[str]
        List of paths to each image mask.
    transform : torchvision.transforms.Compose
        A composition of transformations to apply to the images.

    Methods
    -------
    __len__() -> int
        Returns the number of samples in the dataset.
    __getitem__(index: int) -> Tuple[torch.Tensor]
        Returns the image inside a tuple.
    """

    def __init__(self, images_dir: str, masks_dir: str, image_transform=None, mask_transform=None):
        """
        Parameters
        ----------
        images_dir: str
            Root directory where all scans are located.
        masks_dir: str
            Root directory where all scan masks are located.
        image_transform : torchvision.transforms.Compose, optional
            A composition of transformations to apply to the images (default is None).
        mask_transform : torchvision.transforms.Compose, optional
            A composition of transformations to apply to the masks (default is None).
        """
        self.image_paths = get_file_paths(images_dir, lambda x: x.endswith(".png"))
        self.mask_paths = get_file_paths(masks_dir, lambda x: x.endswith(".png"))
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Number of images and masks do not match. {len(self.image_paths)} images and {len(self.mask_paths)} masks")

        self.image_and_mask_bidict = self._create_image_and_mask_bidict()
        if len(self.image_and_mask_bidict) != len(self.image_paths) + len(self.mask_paths):
            raise ValueError(
                (f"Some images and masks do not match. I.e. a bijection between images and masks does not exist."
                 f"{len(self.image_paths) + len(self.mask_paths)} images and masks, but {len(self.image_and_mask_bidict)} matches."))

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def _create_image_and_mask_bidict(self):
        """Creates a dictionary that maps image paths to the mask path, and masks paths to image paths."""
        image_uids_to_path = {path.split("/")[-1]: path for path in self.image_paths}
        pairs = []
        for mask_path in self.mask_paths:
            uid = mask_path.split("/")[-1]
            image_path = image_uids_to_path[uid]
            pairs.append((image_path, mask_path))

        bidict = {}
        for image_path, mask_path in pairs:
            bidict[image_path] = mask_path
            bidict[mask_path] = image_path

        return bidict

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the image and its label (as an integer) at the specified index.

        Parameters
        ----------
        index : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The image and its corresponding mask
        """
        image_path = self.image_paths[index]
        mask_path = self.image_and_mask_bidict[image_path]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = Image.fromarray(mask)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def save_image_grid(
        images: torch.Tensor, labels: torch.Tensor, label_mapping: Dict[int, str], save_dir: str, grid_size: int = 3
):
    """
    Saves a grid of images with their labels to a specified directory.

    Parameters
    ----------
    images : torch.Tensor
        A tensor containing the images to save in a grid.
    labels : torch.Tensor
        A tensor containing the labels corresponding to the images.
    label_mapping : Dict[int, str]
        A dictionary mapping integer labels back to their string representations.
    save_dir : str
        The directory where the image grid will be saved.
    grid_size : int, optional
        The number of images per row in the grid (default is 3).
    """
    os.makedirs(save_dir, exist_ok=True)
    grid = vutils.make_grid(images, nrow=grid_size, padding=2, normalize=True)
    grid_np = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "image_grid.png"))
    plt.close()


def print_batch_stats(images: torch.Tensor, labels: torch.Tensor, label_mapping: Dict[int, str]):
    """
    Prints statistics about a batch of images and labels.

    Parameters
    ----------
    images : torch.Tensor
        A tensor containing the images in the batch.
    labels : torch.Tensor
        A tensor containing the labels corresponding to the images in the batch.
    label_mapping : Dict[int, str]
        A dictionary mapping integer labels back to their string representations.
    """
    print(f"Batch size: {len(images)}")

    # label_counts = {label_mapping[label.item()]: (labels == label).sum().item() for label in labels.unique()}
    # print("Label distribution:")
    # for label, count in label_counts.items():
    #     print(f"  {label}: {count}")

    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0, 2, 3])
    print(f"Mean pixel values (RGB): {mean.tolist()}")
    print(f"Standard deviation of pixel values (RGB): {std.tolist()}")

    max_value = images.max().item()
    min_value = images.min().item()
    print(f"Max pixel value: {max_value}")
    print(f"Min pixel value: {min_value}")

    dtype = images.dtype
    print(f"Data type: {dtype}")


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    task_heart_images_root = "/bind_tmp/test_write_nii/images"
    task_heart_masks_root = "/bind_tmp/test_write_nii/masks"
    dataset = DecathlonDataset(task_heart_images_root, task_heart_masks_root, transform, transform)
    dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

    images, masks = next(iter(dataloader))

    images_next_to_masks = torch.cat((images, masks), dim=0)

    save_image_grid(images_next_to_masks, None, None, save_dir="/bind_tmp")
    print_batch_stats(images, None, None)
