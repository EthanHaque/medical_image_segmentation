import pytorch_lightning as pl

from medical_image_segmentation.train.model.byol_pytorch import BYOL

from argparse import ArgumentParser
from datetime import datetime
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR10_FFCV", type=str, help="dataset")
    parser.add_argument(
        "--download",
        default=False,
        action="store_true",
        help="wether to download the dataset",
    )
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument(
        "--num_workers",
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
        type=int,
        help="number of workers",
    )
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--base_lr", default=1.0, type=float, help="base learning rate")
    parser.add_argument("--min_lr", default=1e-3, type=float, help="min learning rate")
    parser.add_argument("--linear_loss_weight", default=0.03, type=float, help="weight for the linear loss")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay", default=1.0e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=4096, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--base_momentum", default=0.99, type=float, help="base momentum for byol")
    parser.add_argument("--final_momentum", default=1.0, type=float, help="final momentum for byol")
    parser.add_argument(
        "--comment",
        default=datetime.now().strftime("%b%d_%H-%M-%S"),
        type=str,
        help="wandb comment",
    )
    parser.add_argument("--project", default="essential-byol", type=str, help="wandb project")
    parser.add_argument("--entity", default=None, type=str, help="wandb entity")
    parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
    parser.add_argument(
        "--num_gpus",
        default=int(os.environ.get("SLURM_GPUS_ON_NODE", "4")),
        type=int,
        help="Number of GPUs to use for training",
    )
    parser.add_argument("--max_epochs", default=100, type=int, help="Number of training epochs")

    return parser.parse_args()


def main(args):
    logger = pl.loggers.CSVLogger("logs")

    model = BYOL(**args.__dict__)
    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model)


if __name__ == "__main__":
    args = parse_args()

    main(args)
