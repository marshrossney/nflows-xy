import torch
import torch.nn as nn


from quantum_rotor.transforms import build_sigmoid_module
from quantum_rotor.utils import mod_2pi, as_vector

Tensor = torch.Tensor


class AutoregressiveFlow(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def _forward_v1(self, φ):
        φ0, φ = φ.tensor_split([1], dim=1)
        φ = mod_2pi(φ - φ0)

        θ = torch.zeros_like(φ0)
        ldj = 0.0
        for φ_t, f_t in zip(φ.split(1, dim=1), self.transforms):
            context = as_vector(θ.transpose(1, 2))

            θ_t, ldj_t = f_t(context)(φ_t)

            θ = torch.cat([θ, θ_t], dim=1)
            ldj += ldj_t

        θ = mod_2pi(θ + φ0)

        return θ, ldj

    def _forward_v2(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        φ0, φ = φ.tensor_split([1], dim=1)

        θ = φ0
        ldj = 0.0
        for φ_t, f_t in zip(φ.split(1, dim=1), self.transforms):
            _, θ_tm1 = θ.tensor_split([-1], dim=1)

            context = as_vector((θ - θ_tm1).transpose(1, 2))

            φ_t = mod_2pi(φ_t - θ_tm1)
            θ_t, ldj_t = f_t(context)(φ_t)
            θ_t = mod_2pi(θ_t + θ_tm1)

            θ = torch.cat([θ, θ_t], dim=1)
            ldj += ldj_t

        return θ, ldj

    def _forward_v3(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        φ0, φ = φ.tensor_split([1], dim=1)

        θ = φ0
        ldj = 0.0
        for φ_t, f_t in zip(φ.split(1, dim=1), self.transforms):
            _, θ_tm1 = θ.tensor_split([-1], dim=1)

            U = θ[:, 1:] - θ[:, :-1]

            context = as_vector(U.transpose(1, 2))  # .sum(dim=-1, keepdim=True))

            φ_t = mod_2pi(φ_t - θ_tm1)
            θ_t, ldj_t = f_t(context)(φ_t)
            θ_t = mod_2pi(θ_t + θ_tm1)

            θ = torch.cat([θ, θ_t], dim=1)
            ldj += ldj_t

        return θ, ldj

    def _forward_v4(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        """Same as v3 but just pass the sum of links"""
        φ0, φ = φ.tensor_split([1], dim=1)

        θ = φ0
        ldj = 0.0
        for φ_t, f_t in zip(φ.split(1, dim=1), self.transforms):
            _, θ_tm1 = θ.tensor_split([-1], dim=1)

            U = θ[:, 1:] - θ[:, :-1]

            context = as_vector(U.transpose(1, 2).sum(dim=-1, keepdim=True))

            φ_t = mod_2pi(φ_t - θ_tm1)
            θ_t, ldj_t = f_t(context)(φ_t)
            θ_t = mod_2pi(θ_t + θ_tm1)

            θ = torch.cat([θ, θ_t], dim=1)
            ldj += ldj_t

        return θ, ldj

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        assert φ.shape[1] == len(self.transforms) + 1
        return self._forward_v4(φ)


class AutoregressiveSigmoidFlow(AutoregressiveFlow):
    def __init__(
        self,
        n_lattice: int,
        n_mixture: int,
        net_shape: list[int],
        net_activation: str = "Tanh",
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp_pow: int = 2,
    ):
        transforms = [
            build_sigmoid_module(
                net_shape=net_shape,
                net_activation=net_activation,
                n_mixture=n_mixture,
                weighted=weighted,
                min_weight=min_weight,
                ramp_pow=ramp_pow,
            )
            for _ in range(n_lattice - 1)
        ]
        super().__init__(transforms)
