import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
import torchvision.transforms as transforms

from medical_image_segmentation.analyze_data.pytorch_datasets import DecathlonDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_matmul_precision", default="high", type=str, help="torch matmul precision")

    return parser.parse_args()

def main(args):
    """Entry point for training with PyTorch Lightning."""
    torch.set_float32_matmul_precision(args.torch_matmul_precision)

    logger = pl.loggers.CSVLogger("logs")
    callbacks = [
        RichModelSummary(),
        RichProgressBar(),
    ]

    model = smp.Unet(
        encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
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

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    images_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/images"
    masks_dir = "/scratch/gpfs/RUSTOW/med_datasets/medicaldecathlon/sliced_data/masks"
    decathlon_dataset = DecathlonDataset(images_dir, masks_dir, transform, transform)
    trainer.fit(model, decathlon_dataset)


if __name__ == "__main__":
    args = parse_args()

    main(args)