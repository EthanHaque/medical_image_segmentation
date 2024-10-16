# Adapted from https://github.com/Lightning-Universe/lightning-bolts/blob/748715e50f52c83eb166ce91ebd814cc9ee4f043/src/pl_bolts/callbacks/knn_online.py


import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm


class KNNOnlineEvaluator(Callback):
    """Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ...  # the num of classes in the datamodule
        dm.name = ...  # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        online_eval = KNNOnlineEvaluator(k=100, temperature=0.1)
    """

    def __init__(self, k: int = 200, temperature: float = 0.07, num_classes: int = 1000) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        self.num_classes = num_classes
        self.k = k
        self.temperature = temperature

    def predict(self, query_feature: Tensor, feature_bank: Tensor, target_bank: Tensor) -> Tensor:
        """
        Args:
            query_feature: (B, D) a batch of B query vectors with dim=D
            feature_bank: (N, D) the bank of N known vectors with dim=D
            target_bank: (N, ) the bank of N known vectors' labels

        Returns
        -------
            (B, ) the predicted labels of B query vectors
        """
        dim_b = query_feature.shape[0]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = query_feature @ feature_bank.T
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(target_bank.expand(dim_b, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(dim_b * self.k, self.num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(
            one_hot_label.view(dim_b, -1, self.num_classes) * sim_weight.unsqueeze(dim=-1),
            dim=1,
        )

        # pred_labels
        return pred_scores.argsort(dim=-1, descending=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert not trainer.model.training

        def tqdm_rank_zero_only(iterator, *args, **kwargs):
            if trainer.is_global_zero:
                return tqdm(iterator, *args, **kwargs)
            else:
                return iterator

        # Skip Sanity Check as train_dataloader is not initialized during Sanity Check
        if trainer.train_dataloader is None:
            return

        total_top1 = 0
        total_num = 0
        feature_bank = []
        target_bank = []

        # go through train data to generate feature bank
        for batch in tqdm_rank_zero_only(trainer.train_dataloader, desc="Generating feature bank", leave=False):
            original_images = batch[0]
            labels = batch[1]
            x = original_images.to(pl_module.device)
            target = labels.to(pl_module.device)
            feature = pl_module.forward(x, return_embedding=True)[0].flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            feature_bank.append(feature)
            target_bank.append(target)

        # [N, D]
        feature_bank = torch.cat(feature_bank, dim=0)
        # [N]
        target_bank = torch.cat(target_bank, dim=0)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        # gather representations from other gpus
        if accel.is_distributed:
            feature_bank = concat_all_gather(feature_bank, pl_module)
            target_bank = concat_all_gather(target_bank, pl_module)

        # go through val data to predict the label by weighted knn search
        for batch in tqdm_rank_zero_only(trainer.val_dataloaders, desc="Predicting labels", leave=False):
            images = batch[0]
            labels = batch[1]
            x = images.to(pl_module.device)
            target = labels.to(pl_module.device)
            feature = pl_module.forward(x, return_embedding=True)[0].flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            pred_labels = self.predict(feature, feature_bank, target_bank)

            total_num += x.shape[0]
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        pl_module.log(
            "online_knn_val_acc",
            total_top1 / total_num,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


def concat_all_gather(tensor: Tensor, pl_module: LightningModule) -> Tensor:
    return pl_module.all_gather(tensor).view(-1, *tensor.shape[1:])
