import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Segmentation(pl.LightningModule):
    """Segmentation learner."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=self.hparams.arch,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,  # model output channels (number of classes in your dataset)
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        images, masks = batch

        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)

        return loss
