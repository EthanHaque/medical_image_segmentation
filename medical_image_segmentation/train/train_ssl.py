import os
import torch
from torchvision import models
import torchvision

import numpy as np

from byol_pytorch import BYOL
import pytorch_lightning as pl


from ffcv.loader import Loader, OrderOption
import ffcv

EPOCHS = 2
LR = 3e-4
NUM_GPUS = int(os.environ.get("SLURM_GPUS_ON_NODE", "2"))
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, _):
        print(batch)
        images, labels = batch
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def train_dataloader(self):
        imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
        imagenet_std = np.array([0.229, 0.224, 0.225]) * 255

        image_pipeline_0 = [
            ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((IMAGE_SIZE, IMAGE_SIZE)),
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float16),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        image_pipeline_1 = [
            ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((IMAGE_SIZE, IMAGE_SIZE)),
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float16)
        ]

        label_pipeline = [
            ffcv.fields.basics.IntDecoder(),
            ffcv.transforms.Tensor(),
            ffcv.transforms.Tqueeze(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
        ]

        order = OrderOption.RANDOM if NUM_GPUS > 1 else OrderOption.QUASI_RANDOM
        pipelines = {
            "image": image_pipeline_0,
            "image_0": image_pipeline_1,
            "label": label_pipeline,
        }
        custom_field_mapper = {"image_0": "image"}

        loader = Loader(
            "/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton",
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            order=order,
            os_cache=False,
            drop_last=True,
            pipelines=pipelines,
            distributed=NUM_GPUS > 1,
            custom_field_mapper=custom_field_mapper
        )
        return loader

if __name__ == '__main__':
    resnet = models.resnet50(weights=None)

    model = SelfSupervisedLearner(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )
    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        devices=NUM_GPUS,
        accelerator='gpu',
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        sync_batchnorm=True
    )

    trainer.fit(model)
