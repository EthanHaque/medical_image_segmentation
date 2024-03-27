import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transform_lib
from pytorch_lightning import LightningDataModule
import ffcv
from ffcv.fields.decoders import SimpleRGBImageDecoder, RandomResizedCropRGBImageDecoder, IntDecoder, CenterCropRGBImageDecoder
import numpy as np


import os

DATAMODULE_REGISTRY = {}

def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator

def get_datamodule(name):
    if name in DATAMODULE_REGISTRY:
        return DATAMODULE_REGISTRY[name]
    else:
        raise ValueError(f"No datamodule registered with name {name}")

class BYOLRGBDataTransforms:
    def __init__(
        self, crop_size, mean, std, blur_prob=(1.0, 0.1), solarize_prob=(0.0, 0.2)
    ):
        assert (
            len(blur_prob) == 2 and len(solarize_prob) == 2
        ), "atm only 2 views are supported"
        self.crop_size = crop_size
        self.normalize = transform_lib.Normalize(mean=mean, std=std)
        self.color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.transforms = [
            self.build_transform(bp, sp) for bp, sp in zip(blur_prob, solarize_prob)
        ]

    def build_transform(self, blur_prob, solarize_prob):
        transforms = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.crop_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.RandomApply([self.color_jitter], p=0.8),
                transform_lib.RandomGrayscale(p=0.2),
                transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                transform_lib.RandomSolarize(128, p=solarize_prob),
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                self.normalize,
            ]
        )
        return transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


class BYOLRGBFFCVDataTransforms:
    def __init__(self, device, crop_size, mean, std, solarize_prob=(0.0, 0.2)):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.device = device
        self.transforms = [
            self._build_transforms(sp) for sp in solarize_prob
        ]
    
    def _build_transforms(self, solarize_prob):
        transforms = [
            RandomResizedCropRGBImageDecoder(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.33333)),
            ffcv.transforms.RandomHorizontalFlip(flip_prob=0.5),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            # ffcv.transforms.GaussianBlur(1.0, kernel_size=23),
            ffcv.transforms.RandomSolarization(solarize_prob, 128),
            ffcv.transforms.NormalizeImage(np.array(self.mean) * 255, np.array(self.std) * 255, np.float32),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.Convert(torch.float32),
        ]

        return transforms

    def get_transforms(self):
        return self.transforms

class RGBFFCVDataModule(LightningDataModule):

    def __init__(self, train_path, test_path, batch_size, image_size, num_workers, device, use_distributed, **kwargs):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.device = device
        self.use_distributed = use_distributed

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def std(self):
        raise NotImplementedError()


    def train_dataloader(self):
        image_pipeline_1, image_pipeline_2 = BYOLRGBFFCVDataTransforms(device=self.device, crop_size=self.image_size, mean=self.mean, std=self.std).get_transforms()
        label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze(),]

        pipelines = {
            "image": image_pipeline_1,
            "image_1": image_pipeline_2,
            "label": label_pipeline,
        }
        custom_field_mapper = {"image_1": "image"}

        order = ffcv.loader.OrderOption.QUASI_RANDOM if self.use_distributed else ffcv.loader.OrderOption.RANDOM
        loader = ffcv.loader.Loader(
            self.train_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=order,
            os_cache=True,
            drop_last=True,
            pipelines=pipelines,
            custom_field_mapper=custom_field_mapper
        )
        return loader


    def val_dataloader(self):
        image_pipeline = self.default_transform()

        label_pipeline = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze(),]
        pipelines = {
            "image": image_pipeline,
            "label": label_pipeline,
        }

        order = ffcv.loader.OrderOption.SEQUENTIAL
        loader = ffcv.loader.Loader(
            self.test_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=order,
            os_cache=True,
            drop_last=False,
            pipelines=pipelines,
        )
        return loader

    def default_transform(self):
        mean = np.array(self.mean) * 255
        std = np.array(self.std) * 255
        transform = [
            SimpleRGBImageDecoder(),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.Convert(torch.float32),
            torchvision.transforms.Normalize(mean, std),
        ]
        return transform


@register_datamodule("CIFAR100_FFCV")
class CIFAR100FFCVDataModule(RGBFFCVDataModule):
    NUM_CLASSES = 100
    MEAN = (0.507, 0.487, 0.441)
    STD = (0.268, 0.257, 0.276)
    def __init__(self, batch_size, num_workers, device, use_distributed, **kwargs):
        super().__init__("/scratch/gpfs/eh0560/data/cifar100_ffcv/cifar100_32_train.beton", "/scratch/gpfs/eh0560/data/cifar100_ffcv/cifar100_32_test.beton", batch_size, (32, 32), num_workers, device, use_distributed)

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD


@register_datamodule("CIFAR10_FFCV")
class CIFAR10FFCVDataModule(RGBFFCVDataModule):
    NUM_CLASSES = 10
    MEAN = (0.491, 0.482, 0.447)
    STD = (0.247, 0.243, 0.261)
    def __init__(self, batch_size, num_workers, device, use_distributed, **kwargs):
        super().__init__( "/scratch/gpfs/eh0560/data/cifar10_ffcv/cifar10_32_train.beton", "/scratch/gpfs/eh0560/data/cifar10_ffcv/cifar10_32_test.beton", batch_size, (32, 32), num_workers, device, use_distributed)

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD


@register_datamodule("IMAGENET_FFCV")
class ImageNetFFCVDataModule(RGBFFCVDataModule):
    NUM_CLASSES = 1000
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, batch_size, num_workers, device, use_distributed, **kwargs):
        self.image_size = (112, 112)
        super().__init__( "/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_112_train.beton", "/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_112_test.beton", batch_size, self.image_size, num_workers, device, use_distributed)

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD

    def default_transform(self):
        mean = np.array(self.mean) * 255
        std = np.array(self.std) * 255
        transform = [
            CenterCropRGBImageDecoder(self.image_size, ratio=1.0),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.Convert(torch.float32),
            torchvision.transforms.Normalize(mean, std),
        ]
        return transform
@register_datamodule("IMAGENET")
class ImageNetDataModule(LightningDataModule):
    NUM_CLASSES = 1000
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD

    def prepare_data(self):
        pass

    def setup(self, stage=None):  # called on every GPU
        # build tranforms
        train_transform = BYOLRGBDataTransforms(crop_size=56, mean=self.mean, std=self.std)
        val_transform = self.default_transform()

        # build datasets
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")
        self.train = datasets.ImageFolder(train_data_dir, transform=train_transform)
        self.val = datasets.ImageFolder(val_data_dir, transform=val_transform)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def default_transform(self):
        transform = transform_lib.Compose(
            [
                transform_lib.Resize(112),
                transform_lib.CenterCrop(56),
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                transform_lib.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transform


class CIFARDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

    @property
    def dataset(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def std(self):
        raise NotImplementedError()

    def prepare_data(self):  # called only on 1 GPU
        if self.download:
            self.dataset(self.data_dir, train=True, download=self.download)

    def setup(self, stage=None):  # called on every GPU
        # build tranforms
        train_transform = BYOLRGBDataTransforms(
            crop_size=32,
            mean=self.mean,
            std=self.std,
            blur_prob=[0.0, 0.0],
            solarize_prob=[0.0, 0.2],
        )
        val_transform = self.default_transform()

        # build datasets
        self.train = self.dataset(self.data_dir, train=True, transform=train_transform)
        self.val = self.dataset(self.data_dir, train=False, transform=val_transform)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def default_transform(self):
        transform = transform_lib.Compose(
            [
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                transform_lib.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transform


@register_datamodule("CIFAR10")
class CIFAR10DataModule(CIFARDataModule):
    NUM_CLASSES = 10
    MEAN = (0.491, 0.482, 0.447)
    STD = (0.247, 0.243, 0.261)
    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR10

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD


@register_datamodule("CIFAR100")
class CIFAR100DataModule(CIFARDataModule):
    NUM_CLASSES = 100
    MEAN = (0.507, 0.487, 0.441)
    STD = (0.268, 0.257, 0.276)

    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR100

    @property
    def num_classes(self):
        return self.NUM_CLASSES

    @property
    def mean(self):
        return self.MEAN

    @property
    def std(self):
        return self.STD
