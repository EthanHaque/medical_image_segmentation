import pytorch_lightning as pl
import os
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
import torchvision.transforms as transforms

from medical_image_segmentation.analyze_data.pytorch_datasets import save_image_grid, save_combined_image_grid
from medical_image_segmentation.train.data_loaders.lightning_module import DecathlonHeartDataModule, \
    DecathlonLiverDataModule
from medical_image_segmentation.train.model.segmentation import Segmentation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_matmul_precision", default="high", type=str, help="torch matmul precision")
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--base_lr", default=2.2, type=float, help="base learning rate")
    parser.add_argument("--min_lr", default=1e-2, type=float, help="min learning rate")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay", default=1.0e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--max_epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument(
        "--num_gpus",
        default=int(os.environ.get("SLURM_GPUS_ON_NODE", "4")),
        type=int,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--num_workers",
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
        type=int,
        help="number of workers",
    )
    parser.add_argument("--batch_size", default=512, type=int, help="batch size")
    parser.add_argument("--save_example_predictions", action="store_true", help="Write some images from the prediction step.")

    return parser.parse_args()


def main(args):
    """Entry point for training with PyTorch Lightning."""
    torch.set_float32_matmul_precision(args.torch_matmul_precision)

    logger = pl.loggers.CSVLogger("logs")
    callbacks = [
        RichModelSummary(),
        RichProgressBar(),
    ]

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )

    n_classes = 1
    images_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/images"
    masks_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/masks"
    split_file = (
        "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/Task04_hippocampus/split_10_percent_train.json"
    )
    decathlon_dataset = DecathlonLiverDataModule(
        images_dir,
        masks_dir,
        split_file,
        args.batch_size,
        args.num_workers,
    )

    model = Segmentation(n_classes, **args.__dict__)
    trainer.fit(model, decathlon_dataset)

    trainer.test(model, decathlon_dataset)

    if args.save_example_predictions:
        preds = trainer.predict(model, decathlon_dataset)
        first_prediction = preds[0]
        images, pred_masks, true_masks = first_prediction
        save_combined_image_grid(
            images,
            pred_masks,
            true_masks,
            "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/images",
            grid_size=16,
        )
        save_image_grid(pred_masks, "/scratch/gpfs/eh0560/repos/medical-image-segmentation/data/images", grid_size=16)


if __name__ == "__main__":
    args = parse_args()

    main(args)
