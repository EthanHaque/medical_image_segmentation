from typing import Tuple, List

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from medical_image_segmentation.train.optimizer.lars import LARS
from medical_image_segmentation.train.scheduler.cosine_annealing import LinearWarmupCosineAnnealingLR


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1.
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_coeff
        return dice_loss


def post_process_masks(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    masks = (probs > threshold).float()
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

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = LARS(
            params,
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def loss(self, logits, masks):
        loss_fn = DiceLoss()
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
        logits = self(images)
        predicted_masks = post_process_masks(logits)
        return images, predicted_masks, masks
