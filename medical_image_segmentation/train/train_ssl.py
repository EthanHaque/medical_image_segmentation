import argparse
import os
import torch
import torchvision

import numpy as np

from byol_pytorch import BYOL
import pytorch_lightning as pl


from ffcv.loader import Loader, OrderOption
import ffcv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Supervised Learning with BYOL and PyTorch")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), help="Number of workers for data loading")
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", "2")), help="Number of GPUs for training")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file to restore training")

    return parser.parse_args()

args = parse_args()

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, _):
        images_0 = batch[0]
        images_1 = batch[2]
        images = torch.cat((images_0, images_1), dim=0)
        loss = self.forward(images)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.lr)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def train_dataloader(self):
        imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
        imagenet_std = np.array([0.229, 0.224, 0.225]) * 255

        image_pipeline_0 = [
            ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((args.image_size, args.image_size)),
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        image_pipeline_1 = [
            ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((args.image_size, args.image_size)),
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(imagenet_mean, imagenet_std, np.float32)
        ]

        label_pipeline = [
            ffcv.fields.basics.IntDecoder(),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.Squeeze(),
            ffcv.transforms.ToDevice(self.trainer.local_rank, non_blocking=True),
        ]

        order = OrderOption.RANDOM if args.num_gpus > 1 else OrderOption.QUASI_RANDOM
        pipelines = {
            "image": image_pipeline_0,
            "image_0": image_pipeline_1,
            "label": label_pipeline,
        }
        custom_field_mapper = {"image_0": "image"}

        loader = Loader(
            "/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            order=order,
            os_cache=True,
            drop_last=True,
            pipelines=pipelines,
            distributed=args.num_gpus > 1,
            custom_field_mapper=custom_field_mapper
        )
        return loader

if __name__ == '__main__':
    resnet = torchvision.models.resnet18(weights=None)

    model = SelfSupervisedLearner(
        resnet,
        image_size=args.image_size,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )
    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        devices=args.num_gpus,
        accelerator='gpu',
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
    )

    trainer.fit(model, ckpt_path=args.checkpoint_path)
