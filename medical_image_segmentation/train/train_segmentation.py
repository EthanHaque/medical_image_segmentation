import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
import torchvision.transforms as transforms

from medical_image_segmentation.analyze_data.pytorch_datasets import DecathlonDataset
from medical_image_segmentation.train.model.segmentation import Segmentation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_matmul_precision", default="high", type=str, help="torch matmul precision")
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--lr", default=4e-3, type=float, help="base learning rate")

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

    model = Segmentation(**args.__dict__)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    images_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/images"
    masks_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/masks"
    decathlon_dataset = DecathlonDataset(images_dir, masks_dir, transform, transform)
    trainer.fit(model, decathlon_dataset)


if __name__ == "__main__":
    args = parse_args()

    main(args)