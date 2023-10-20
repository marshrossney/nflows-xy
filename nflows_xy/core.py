from math import pi as π
from typing import NamedTuple

import torch
import torch.nn as nn

from nflows_xy.xy import Action
from nflows_xy.flows import Flow

Tensor = torch.Tensor


class PullbackAction(Action):
    def __init__(self, flow: nn.Module, target: Action):
        super().__init__(target.beta, target.lattice_size, target.lattice_dim)
        self.flow = flow
        self.target = target

    @torch.no_grad()
    def __call__(self, u: Tensor) -> Tensor:
        φ, ldj = self.flow(u)
        return self.target(φ) - ldj

    @torch.enable_grad()
    def grad(self, u: Tensor) -> Tensor:
        u.requires_grad_(True)
        u.grad = None
        φ, ldj = self.flow(u)

        action = self.target(φ) - ldj

        (gradient,) = torch.autograd.grad(
            outputs=action,
            inputs=u,
            grad_outputs=torch.ones_like(action),
        )

        u.requires_grad_(False)
        u.grad = None

        return gradient


class FlowBasedSampler(nn.Module):
    class Fields(NamedTuple):
        inputs: Tensor
        outputs: Tensor

    class Actions(NamedTuple):
        base: Tensor
        target: Tensor
        pushforward: Tensor
        pullback: Tensor

    def __init__(self, flow: Flow, target: Action):
        super().__init__()
        self.target = target
        self.register_module("flow", flow)
        self.register_buffer(
            "_dummy_buffer", torch.tensor(0.0), persistent=False
        )

    def base_sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        u = self._dummy_buffer.new_empty(
            (batch_size, *self.target.lattice_shape, 1)
        ).uniform_(0, 2 * π)
        return u, u.new_zeros(batch_size, 1)

    def forward(self, batch_size: int) -> tuple[Tensor, Tensor]:
        u, S_base = self.base_sample(batch_size)
        φ, ldj = self.flow(u)
        S_target = self.target(φ)

        S_pushforward = S_base + ldj
        S_pullback = S_target - ldj

        fields = self.Fields(inputs=u, outputs=φ)
        actions = self.Actions(S_base, S_target, S_pushforward, S_pullback)

        return fields, actions
