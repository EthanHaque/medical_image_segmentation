import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        return optimizer

    def loss(self, logits, masks):
        loss_fn = nn.BCEWithLogitsLoss()
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
        print(f"logits max {logits.max()}")
        predicted_masks = post_process_masks(logits)
        print(f"mask max {predicted_masks.max()}")
        return images, predicted_masks, masks
