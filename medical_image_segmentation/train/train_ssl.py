import argparse
import os
import torch
import torchvision

import numpy as np

from byol_pytorch import BYOL
import pytorch_lightning as pl


from ffcv.loader import Loader, OrderOption
import ffcv

from callback.knn import KNNOnlineEvaluator
from medical_image_segmentation.train.data_loaders.ffcv_loader import create_train_loader_ssl, create_val_loader_ssl

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Supervised Learning with BYOL and PyTorch")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")), help="Number of workers for data loading")
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", "4")), help="Number of GPUs for training")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file to restore training")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of epochs to warm up to set learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--dry", action="store_true", help="Dry run")
    parser.add_argument("--float_matmul_precision", type=str, default="highest", help="Setting float32 matrix multiplication precision. See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_preci")
    parser.add_argument("--train_subset_size", type=int, required=False)
    parser.add_argument("--val_subset_size", type=int, required=False)

    return parser.parse_args()

# this is pretty awful
args = parse_args()


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images, return_embedding=False):
        return self.learner(images, return_embedding=return_embedding)

    def on_train_start(self):
        self.logger.log_hyperparams(args)

    def training_step(self, batch, _):
        # original_images = batch[0]
        # labels = batch[1]
        images_aug_1 = batch[2]
        images_aug_2 = batch[3]
        images = torch.cat((images_aug_1, images_aug_2), dim=0)
        loss = self.forward(images)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {'loss': loss}

    def validation_step(self, batch, _):
        return

    def configure_optimizers(self):
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented as option")

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epochs])

        return [optimizer], [scheduler]


    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def train_dataloader(self):
        subset_size = args.train_subset_size if args.train_subset_size else -1
        loader = create_train_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_train.beton",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            num_gpus=args.num_gpus,
            in_memory=True,
            subset_size=subset_size,
        )

        return loader

    def val_dataloader(self):
        subset_size = args.val_subset_size if args.val_subset_size else -1
        loader = create_val_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/imagenet_ffcv/imagenet_val.beton",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            num_gpus=args.num_gpus,
            in_memory=True,
            subset_size=subset_size,
        )

        return loader


def setup_train_objects():
    """Creates objects for training."""
    resnet = torchvision.models.resnet18(weights=None)

    model = SelfSupervisedLearner(
        resnet,
        image_size=args.image_size,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    )

    logger = pl.loggers.CSVLogger("logs")
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        KNNOnlineEvaluator(num_classes=1000),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch", log_momentum=True, log_weight_decay=True)
    ]

    trainer = pl.Trainer(
        strategy='ddp_find_unused_parameters_true',
        devices=args.num_gpus,
        accelerator='gpu',
        max_epochs=args.epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        callbacks=callbacks,
        logger=logger,
    )

    return model, trainer


def main():
    """Dispatches to correct method calls based on args"""
    torch.set_float32_matmul_precision(args.float_matmul_precision)
    model, trainer = setup_train_objects()

    if args.dry:
        return

    if args.checkpoint_path:
        trainer.fit(model, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
