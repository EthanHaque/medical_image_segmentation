import ffcv
from ffcv.loader import Loader, OrderOption
import numpy as np
import torchvision


def create_train_loader_ssl(this_device: str, beton_file_path: str, batch_size: int, num_workers: int,
                        image_size: int, num_gpus: int, in_memory: bool, subset_size: int) -> Loader:

    imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
    imagenet_std = np.array([0.229, 0.224, 0.225]) * 255

    image_pipeline_0 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((image_size, image_size)),
        ffcv.transforms.RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(this_device, non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
    ]

    image_pipeline_1 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((image_size, image_size)),
        ffcv.transforms.RandomHorizontalFlip(),
        ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        ffcv.transforms.RandomGrayscale(0.2),
        ffcv.transforms.RandomSolarization(0.2, 128),
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
        "image": image_pipeline_0,
        "image_0": image_pipeline_1,
        "label": label_pipeline,
    }
    custom_field_mapper = {"image_0": "image"}

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
        ffcv.fields.rgb_image.CenterCropRGBImageDecoder((image_size, image_size)),
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
