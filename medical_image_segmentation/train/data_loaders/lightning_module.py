import torch
from torchvision import datasets
from torchvision.transforms import v2 as transform_lib
from pytorch_lightning import LightningDataModule
import ffcv


from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder

import os


class BYOLRGBDataTransforms(torch.nn.Module):
    def __init__(
            self, crop_size, mean, std, blur_prob=(1.0, 0.1), solarize_prob=(0.0, 0.2)
    ):
        super().__init__()
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


class CIFAR100FFCVDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 100

    @property
    def mean(self):
        return (0.507, 0.487, 0.441)

    @property
    def std(self):
        return (0.268, 0.257, 0.276)

    def train_dataloader(self):
        train_transforms_1 = [
            transform_lib.RandomResizedCrop(32),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.RandomApply([transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transform_lib.RandomGrayscale(p=0.2),
            transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=1.0),
            transform_lib.RandomSolarize(128, p=0.0),
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.float32, scale=True),
            transform_lib.Normalize(mean=self.mean, std=self.std)
        ]


        train_transforms_2 = [
            transform_lib.RandomResizedCrop(32),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.RandomApply([transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transform_lib.RandomGrayscale(p=0.2),
            transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=0.1),
            transform_lib.RandomSolarize(128, p=0.2),
            transform_lib.ToImage(),
            transform_lib.ToDtype(torch.float32, scale=True),
            transform_lib.Normalize(mean=self.mean, std=self.std)
        ]

        image_pipeline = [ SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToTorchImage()] + train_transforms_1
        image_pipeline_1 = [ SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToTorchImage()] + train_transforms_2
        label_pipeline = [ IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze(), ]
        pipelines = {
            "image": image_pipeline,
            "image_1": image_pipeline_1,
            "label": label_pipeline,
        }
        custom_field_mapper = {"image_1": "image"}
        loader = ffcv.loader.Loader(
            self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=ffcv.loader.OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines=pipelines,
            custom_field_mapper=custom_field_mapper
        )
        return loader

    def val_dataloader(self):
        val_transform = [
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                transform_lib.Normalize(mean=self.mean, std=self.std),
            ]
        image_pipeline = [ SimpleRGBImageDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.ToTorchImage()] + val_transform
        label_pipeline = [ IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze(), ]
        loader = ffcv.loader.Loader(
            self.data_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=ffcv.loader.OrderOption.SEQUENTIAL,
            os_cache=True,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline}
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


class ImageNetDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 1000

    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def std(self):
        return (0.229, 0.224, 0.225)

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


class CIFAR10DataModule(CIFARDataModule):
    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR10

    @property
    def num_classes(self):
        return 10

    @property
    def mean(self):
        return (0.491, 0.482, 0.447)

    @property
    def std(self):
        return (0.247, 0.243, 0.261)


class CIFAR100DataModule(CIFARDataModule):
    def __init__(self, data_dir, batch_size, num_workers, download, **kwargs):
        super().__init__(data_dir, batch_size, num_workers, download)

    @property
    def dataset(self):
        return datasets.CIFAR100

    @property
    def num_classes(self):
        return 100

    @property
    def mean(self):
        return (0.507, 0.487, 0.441)

    @property
    def std(self):
        return (0.268, 0.257, 0.276)
