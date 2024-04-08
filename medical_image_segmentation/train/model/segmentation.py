from typing import Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def post_process_masks(logits):
    probs = torch.softmax(logits, dim=1)  # Assuming channel 0 is for background
    masks = torch.argmax(probs, dim=1)  # Take the argmax across channel dimension
    return masks



class Segmentation(pl.LightningModule):
    """Segmentation learner."""

    def __init__(self, n_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=self.hparams.arch,
            encoder_weights=None,
            in_channels=1,
            classes=n_classes,
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        return optimizer

    def loss(self, logits, masks):
        print(f"logits shape {logits.shape}")
        print(f"masks shape {masks.shape}")
        print(f"logits dtype {logits.dtype}")
        print(f"masks dtype {masks.dtype}")
        assert logits.dtype.is_floating_point
        assert masks.dtype == torch.long
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, masks)
        return loss

    def training_step(self, batch, batch_idx):
        images, masks = batch

        logits = self.forward(images)
        loss = self.loss(logits, masks)

        metric_log = {
            "train/loss": loss
        }

        self.log_dict(
            metric_log,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        logits = self.forward(images)
        loss = self.loss(logits, masks)

        metric_log = {
            "val/loss": loss
        }

        self.log_dict(
            metric_log,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.forward(images)

        predicted_masks = post_process_masks(logits)
        return images, predicted_masks, masks