# Adapted from https://github.com/Lightning-Universe/lightning-bolts/blob/748715e50f52c83eb166ce91ebd814cc9ee4f043/src/pl_bolts/callbacks/ssl_online.py#L17

from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
import pytorch_lightning as pl
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Optimizer
from torchmetrics.functional import accuracy



class SSLLinearEval(Callback):  # pragma: no cover
    """Passes the representation from the model to a linear layer for classification."""

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
        """
        super().__init__()

        self.z_dim = z_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[nn.Sequential] = None
        self.num_classes: Optional[int] = num_classes

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = nn.Sequential(Flatten(),
                                              nn.Dropout(p=self.drop_p),
                                              nn.Linear(self.z_dim, self.num_classes, bias=True)
                                              ).to(pl_module.device)

        # switch fo PL compatibility reasons
        strategy = trainer.strategy
        if strategy is not None and strategy.is_distributed:
            if isinstance(strategy, pl.strategies.DDPStrategy):
                from torch.nn.parallel import DistributedDataParallel

                self.online_evaluator = DistributedDataParallel(self.online_evaluator, device_ids=[pl_module.device])
            elif isinstance(strategy, pl.strategies.DPStrategy):
                from torch.nn.parallel import DataParallel

                self.online_evaluator = DataParallel(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed strategy. The online evaluator will not sync."
                )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad(), set_training(pl_module, False):
            x, y = self.to_device(batch, pl_module.device)
            print(pl_module.forward(x, return_embedding=True))
            representations = pl_module.forward(x, return_embedding=True).flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        acc = accuracy(mlp_logits.softmax(-1), y, task="multiclass", num_classes=self.num_classes)

        return acc, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        train_acc, mlp_loss = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        val_acc, mlp_loss = self.shared_step(pl_module, batch)
        pl_module.log("online_val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def state_dict(self) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._recovered_callback_state = state_dict


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).

    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)

class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)