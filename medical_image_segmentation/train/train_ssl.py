import argparse
import os
import torch
import torchvision

import numpy as np

from byol_pytorch import BYOL
import pytorch_lightning as pl


from ffcv.loader import Loader, OrderOption
import ffcv

from tqdm import tqdm

from medical_image_segmentation.train.callback.linear_eval import SSLLinearEval
from medical_image_segmentation.train.data_loaders.ffcv_loader import create_train_loader_ssl, create_val_loader_ssl
from medical_image_segmentation.train.optimizer.lars import LARS
from medical_image_segmentation.train.scheduler.cosine_annealing import LinearWarmupCosineAnnealingLR

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Supervised Learning with BYOL and PyTorch")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-3, help="Minimum learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--num_workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")), help="Number of workers for data loading")
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SLURM_GPUS_ON_NODE", "4")), help="Number of GPUs for training")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file to restore training")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of epochs to warm up to set learning rate")
    parser.add_argument("--optimizer", type=str, default="lars", help="Optimizer to use")
    parser.add_argument("--dry", action="store_true", help="Dry run")
    parser.add_argument("--float_matmul_precision", type=str, default="highest", help="Setting float32 matrix multiplication precision. See https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_preci")
    parser.add_argument("--train_subset_size", type=int, required=False)
    parser.add_argument("--val_subset_size", type=int, required=False)
    parser.add_argument("--val_every", type=int, default=1, help="Runs validation step every n epochs")
    parser.add_argument("--run_single_validation", action="store_true", help="Run a single validation step.")

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
            optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
        elif args.optimizer == "lars":
            optimizer = LARS(self.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented as option")

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            warmup_start_lr=args.min_lr,
            eta_min=args.min_lr
        )

        return [optimizer], [scheduler]


    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def train_dataloader(self):
        subset_size = args.train_subset_size if args.train_subset_size else -1
        loader = create_train_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/cifar10_ffcv/cifar10_train.beton",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            num_gpus=args.num_gpus,
            in_memory=True,
            subset_size=subset_size,
        )

        def tqdm_rank_zero_only(iterator, *args, **kwargs):
            if self.trainer.is_global_zero:
                return tqdm(iterator, *args, **kwargs)
            else:
                return iterator

        for _ in tqdm_rank_zero_only(loader, desc="Prefetching train data"):
            pass

        return loader

    def val_dataloader(self):
        subset_size = args.val_subset_size if args.val_subset_size else -1
        loader = create_val_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/cifar10_ffcv/cifar10_train.beton",
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
    resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

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
        pl.callbacks.LearningRateMonitor(logging_interval="epoch", log_momentum=True, log_weight_decay=True),
        SSLLinearEval(256, drop_p=0.2, num_classes=1000),
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
        check_val_every_n_epoch=args.val_every,
        log_every_n_steps=1
    )

    return model, trainer


def main():
    """Dispatches to correct method calls based on args"""
    torch.set_float32_matmul_precision(args.float_matmul_precision)
    model, trainer = setup_train_objects()

    if args.dry:
        return

    if args.run_single_validation:
        if args.checkpoint_path:
            trainer.validate(model, ckpt_path=args.checkpoint_path)
        else:
            print("Please provide a checkpoint path using --checkpoint_path to run a single validation step.")
        return

    if args.checkpoint_path:
        trainer.fit(model, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
