from math import pi as π
from typing import NamedTuple

import torch
import torch.nn as nn

from quantum_rotor.flows import Flow
from quantum_rotor.action import ActionPBC

Tensor = torch.Tensor


class FlowBasedModel(nn.Module):
    class Fields(NamedTuple):
        inputs: Tensor
        outputs: Tensor

    class Actions(NamedTuple):
        base: Tensor
        target: Tensor
        pushforward: Tensor
        pullback: Tensor

    def __init__(self, flow: Flow, n_lattice: int, beta: float):
        super().__init__()
        self.n_lattice = n_lattice
        self.beta = beta
        self.target = ActionPBC(beta)
        self.register_module("flow", flow)
        self.register_buffer(
            "_dummy_buffer", torch.tensor(0.0), persistent=False
        )

    def base_sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        u = self._dummy_buffer.new_empty(
            (batch_size, self.n_lattice, 1)
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
