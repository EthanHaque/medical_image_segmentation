import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from medical_image_segmentation.train.optimizer.lars import LARS
from medical_image_segmentation.train.scheduler.cosine_annealing import LinearWarmupCosineAnnealingLR
from medical_image_segmentation.train.data_loaders.ffcv_loader import create_train_loader_ssl, create_val_loader_ssl
from tqdm import tqdm


import math

from torchvision import models


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, arch, hidden_dim, proj_dim, low_res):
        super().__init__()

        # backbone
        self.encoder = models.__dict__[arch]()
        self.feat_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()

        # modify the encoder for lower resolution
        if low_res:
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self._reinit_all_layers()

        # build heads
        self.projection = MLP(self.feat_dim, hidden_dim, proj_dim)

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.projection(feats)
        return z, feats


class BYOL(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.current_momentum = self.hparams.base_momentum

        # online encoder
        self.online_encoder = Encoder(
            arch=self.hparams.arch,
            hidden_dim=self.hparams.hidden_dim,
            proj_dim=self.hparams.proj_dim,
            low_res='CIFAR' in self.hparams.dataset)

        # momentum encoder
        self.momentum_encoder = Encoder(
            arch=self.hparams.arch,
            hidden_dim=self.hparams.hidden_dim,
            proj_dim=self.hparams.proj_dim,
            low_res='CIFAR' in self.hparams.dataset)
        self.initialize_momentum_encoder()

        # predictor
        self.predictor = MLP(
            input_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.proj_dim)

        # linear layer for eval
        self.linear = torch.nn.Linear(
            self.online_encoder.feat_dim, self.hparams.num_classes)

    @torch.no_grad()
    def initialize_momentum_encoder(self):
        params_online = self.online_encoder.parameters()
        params_momentum = self.momentum_encoder.parameters()
        for po, pm in zip(params_online, params_momentum):
            pm.data.copy_(po.data)
            pm.requires_grad = False

    def collect_params(self, models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                if exclude_bias_and_bn and any(
                        s in name for s in ['bn', 'downsample.1', 'bias']):
                    param_dict = {
                        'params': param,
                        'weight_decay': 0.,
                        'lars_exclude': True}
                    # NOTE: with the current pytorch lightning bolts
                    # implementation it is not possible to exclude 
                    # parameters from the LARS adaptation
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def configure_optimizers(self):
        params = self.collect_params([
            self.online_encoder, self.predictor, self.linear])
        optimizer = LARS(params,
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.linear(self.online_encoder.encoder(x))

    def cosine_similarity_loss(self, preds, targets):
        preds = F.normalize(preds, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        return 2 - 2 * (preds * targets).sum(dim=-1).mean()

    def training_step(self, batch, batch_idx):
        # original_images = batch[0]
        labels = batch[1]
        view_1 = batch[2]
        view_2 = batch[3]
        views = [view_1, view_2]

        # forward online encoder
        input_online = torch.cat(views, dim=0)
        z, feats = self.online_encoder(input_online)
        preds = self.predictor(z)

        # forward momentum encoder
        with torch.no_grad():
            input_momentum = torch.cat(views[::-1], dim=0)
            targets, _ = self.momentum_encoder(input_momentum)

        # compute BYOL loss
        loss = self.cosine_similarity_loss(preds, targets)

        # train linear layer
        preds_linear = self.linear(feats.detach())
        loss_linear = F.cross_entropy(preds_linear, labels.repeat(2))

        # gather results and log stats
        logs = {
            'loss': loss,
            'loss_linear': loss_linear,
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'momentum': self.current_momentum}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss + loss_linear * self.hparams.linear_loss_weight

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update momentum encoder
        self.momentum_update(
            self.online_encoder, self.momentum_encoder, self.current_momentum)
        # update momentum value
        max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        self.current_momentum = self.hparams.final_momentum - \
                                (self.hparams.final_momentum - self.hparams.base_momentum) * \
                                (math.cos(math.pi * self.trainer.global_step / max_steps) + 1) / 2

    def train_dataloader(self):
        loader = create_train_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/cifar100_ffcv/cifar100_train.beton",
            batch_size=256,
            num_workers=8,
            image_size=32,
            num_gpus=1,
            in_memory=True,
            subset_size=-1,
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
        loader = create_val_loader_ssl(
            this_device=self.trainer.local_rank,
            beton_file_path="/scratch/gpfs/eh0560/data/cifar100_ffcv/cifar100_test.beton",
            batch_size=256,
            num_workers=8,
            image_size=32,
            num_gpus=1,
            in_memory=True,
            subset_size=-1,
        )

        return loader

    @torch.no_grad()
    def momentum_update(self, online_encoder, momentum_encoder, m):
        online_params = online_encoder.parameters()
        momentum_params = momentum_encoder.parameters()
        for po, pm in zip(online_params, momentum_params):
            pm.data.mul_(m).add_(po.data, alpha=1. - m)

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]

        # predict using online encoder
        preds = self(images)

        # calculate accuracy @k
        acc1, acc5 = self.accuracy(preds, labels)

        # gather results and log
        logs = {'val/acc@1': acc1, 'val/acc@5': acc5}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def accuracy(self, preds, targets, k=(1, 5)):
        preds = preds.topk(max(k), 1, True, True)[1].t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))

        res = []
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res