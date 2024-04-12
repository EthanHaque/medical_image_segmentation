import ffcv
from ffcv.loader import Loader
from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from medical_image_segmentation.analyze_data.pytorch_datasets import DecathlonDataset


def compute_mean_and_std(loader):
    """Computes the mean and standard deviation of a dataset given a loader."""
    sum_ = 0
    sum_squared = 0
    num_pixels = 0

    for batch in tqdm(loader):
        images = batch[0].to(torch.float64)
        sum_ += torch.sum(images, dim=[0, 2, 3])  # Sum over batch, height, and width
        sum_squared += torch.sum(images**2, dim=[0, 2, 3])
        num_pixels += images.size(0) * images.size(2) * images.size(3)  # Batch size * Height * Width

    mean = sum_ / num_pixels
    mean_of_squares = sum_squared / num_pixels
    mean_squared = mean**2
    std = torch.sqrt(mean_of_squares - mean_squared)

    return mean, std


def compute_mean_and_std_pytorch(dataset: Dataset, batch_size=8, num_workers=2):
    """Computes mean and standard deviation for a PyTorch dataset."""

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return compute_mean_and_std(loader)


def compute_mean_and_std_ffcv(beton_file_path, batch_size=8, num_workers=2):
    """Computes mean and standard deviation for a dataset stored in a .beton file."""
    loader = Loader(
        beton_file_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=ffcv.loader.OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=False,
        pipelines={
            "image": [
                CenterCropRGBImageDecoder((224, 224), 1.0),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToTorchImage(),
            ],
            "label": [IntDecoder()],
        },
        distributed=False,
    )

    return compute_mean_and_std(loader)


if __name__ == "__main__":
    images_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/images"
    masks_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/masks"
    split_file = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/split_all_in_train.json"
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = DecathlonDataset(images_dir, masks_dir, 1, transform, transform, "train", split_file)
    mean, std = compute_mean_and_std_pytorch(dataset)
    print(f"Mean: {mean}, Std: {std}")
