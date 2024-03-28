"""
References
----------
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/pytorch/pytorch/blob/1.6/torch/optim/sgd.py
    - https://github.com/Lightning-Universe/lightning-bolts/blob/748715e50f52c83eb166ce91ebd814cc9ee4f043/src/pl_bolts/optimizers/lars.py#L13
"""

import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """
    Extends SGD in PyTorch with LARS scaling from the paper `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float
        Learning rate.
    momentum : float, optional
        Momentum factor.
    weight_decay : float, optional
        Weight decay (L2 penalty).
    dampening : float, optional
        Dampening for momentum.
    nesterov : bool, optional
        Enables Nesterov momentum.
    trust_coefficient : float, optional
        Trust coefficient for computing LR.
    eps : float, optional
        Eps for division denominator.


    Example
    -------
    >>> model = torch.nn.Linear(10, 1)
    >>> input = torch.Tensor(10)
    >>> target = torch.Tensor([1.])
    >>> loss_fn = lambda input, target: (input - target) ** 2
    >>>
    >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

    Notes
    -----
    The application of momentum in the SGD part is modified according to
    the PyTorch standards. LARS scaling fits into the equation in the
    following fashion.

    .. math::
        \begin{aligned}
            g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
            v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
            p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
        \end{aligned}

    where :math:`p`, :math:`g`, :math:`v`, :math:`\mu` and :math:`\beta` denote the
    parameters, gradient, velocity, momentum, and weight decay respectively.
    The :math:`lars_lr` is defined by Eq. 6 in the paper.
    The Nesterov version is analogously modified.

    Warnings
    --------
    Parameters with weight decay set to 0 will automatically be excluded from
    layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
    and BYOL.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
    ) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "trust_coefficient": trust_coefficient,
            "eps": eps,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Load state."""
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure: callable, optional
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                    lars_lr *= group["trust_coefficient"]

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
