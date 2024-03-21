import ffcv
from ffcv.loader import Loader, OrderOption
import numpy as np
import torchvision
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_train_loader_ssl(this_device: str, beton_file_path: str, batch_size: int, num_workers: int,
                        image_size: int, num_gpus: int, in_memory: bool, subset_size: int) -> Loader:

    imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
    imagenet_std = np.array([0.229, 0.224, 0.225]) * 255

    image_pipeline = [
        # setting ratio to 1.0 takes the largest portion of the original image
        ffcv.fields.rgb_image.CenterCropRGBImageDecoder((image_size, image_size), ratio=1.0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32),
    ]

    image_pipeline_1 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((image_size, image_size)),
        ffcv.transforms.RandomHorizontalFlip(),
        # ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        # ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32),
        # torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
    ]

    image_pipeline_2 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((image_size, image_size)),
        ffcv.transforms.RandomHorizontalFlip(),
        # ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        # ffcv.transforms.RandomGrayscale(0.2),
        # ffcv.transforms.RandomSolarization(0.2, 128),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32)
    ]

    label_pipeline = [
        ffcv.fields.basics.IntDecoder(),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.Squeeze(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
    ]

    order = OrderOption.RANDOM if num_gpus > 1 else OrderOption.QUASI_RANDOM
    pipelines = {
        "image": image_pipeline,
        "image_1": image_pipeline_1,
        "image_2": image_pipeline_2,
        "label": label_pipeline,
    }
    custom_field_mapper = {"image_1": "image", "image_2": "image"}

    if subset_size > 0:
        loader = Loader(
            beton_file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines=pipelines,
            distributed=num_gpus > 1,
            custom_field_mapper=custom_field_mapper,
            indices=list(range(subset_size)),
        )
    else:
        loader = Loader(
            beton_file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines=pipelines,
            distributed=num_gpus > 1,
            custom_field_mapper=custom_field_mapper,
        )

    return loader


def create_val_loader_ssl(this_device: str, beton_file_path: str, batch_size: int, num_workers: int,
                        image_size: int, num_gpus: int, in_memory: bool, subset_size: int) -> Loader:

    imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
    imagenet_std = np.array([0.229, 0.224, 0.225]) * 255

    image_pipeline = [
        # setting ratio to 1.0 takes the largest portion of the original image
        ffcv.fields.rgb_image.CenterCropRGBImageDecoder((image_size, image_size), ratio=1.0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32),
    ]

    label_pipeline = [
        ffcv.fields.basics.IntDecoder(),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.Squeeze(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
    ]

    order = OrderOption.SEQUENTIAL
    pipelines = {
        "image": image_pipeline,
        "label": label_pipeline,
    }

    if subset_size > 0:
        loader = Loader(
            beton_file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines=pipelines,
            distributed=num_gpus > 1,
            indices=list(range(subset_size)),
        )
    else:
        loader = Loader(
            beton_file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines=pipelines,
            distributed=num_gpus > 1,
        )

    return loader

if __name__ == "__main__":
    loader = create_train_loader_ssl(
        torch.device("cuda:0"),
        "/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton",
        batch_size=4,
        num_workers=1,
        image_size=56,
        num_gpus=1,
        in_memory=False,
        subset_size=16,
    )

    images = [[], [], []]
    for (x1, y, x2, x3) in tqdm(loader):
        for i, x in enumerate([x1, x2, x3]):
            images[i].append(
                torchvision.utils.make_grid(
                    x,
                    normalize=True,
                    scale_each=True,
                    nrow=4,
                )
            )
        if len(images[-1]) == 3:
            break

    for i in range(3):
        images[i] = torch.cat(images[i], 2)
        images[i] -= images[i].min()
        images[i] /= images[i].max()
        images[i] *= 255
        images[i] = images[i].cpu().numpy().transpose(1, 2, 0).astype("int")

    images = np.concatenate(images, 0)

    fig, axs = plt.subplots(1, 1, figsize=(9, 3))
    axs.imshow(images, extent=[0, 1, 0, 1], interpolation="nearest", aspect="auto")

    axs.plot([0.33, 0.33], [0, 1], c="red", linewidth=3)
    axs.plot([0.67, 0.67], [0, 1], c="red", linewidth=3)
    axs.plot([0, 1], [0.67, 0.67], c="green", linewidth=3)
    axs.plot([0, 1], [0.33, 0.33], c="green", linewidth=3)

    axs.set_xticks([])
    axs.set_yticks([])
    for i, pos in enumerate([0.15, 0.45, 0.75]):
        fig.text(
            0.02,
            pos,
            f"view {3 - i}",
            rotation=90,
            color="green",
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )
    for i, pos in enumerate([0.22, 0.53, 0.84]):
        fig.text(
            pos,
            0.93,
            f"batch {i + 1}",
            color="red",
            fontweight="bold",
            horizontalalignment="center",
        )
    plt.subplots_adjust(0.03, 0.0, 1, 0.9)
    plt.show()