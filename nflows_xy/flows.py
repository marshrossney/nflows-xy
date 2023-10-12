from abc import ABC, abstractmethod
from itertools import cycle

import torch
import torch.nn as nn

from nflows_xy.transforms import build_sigmoid_module
from nflows_xy.utils import mod_2pi, as_vector

Tensor = torch.Tensor


class Flow(nn.Module, ABC):
    @abstractmethod
    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        ...


class DummyFlow(Flow):
    def __init__(self):
        super().__init__()
        self.register_parameter("phase", nn.Parameter(torch.tensor(0.0)))

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        θ = self.phase
        φ = mod_2pi(u + θ)
        return φ, φ.new_zeros(φ.shape[0], 1)


class AutoregressiveFlow(Flow):
    """One-dimensional autoregressive flow model.

    Args:
        lattice_size: Number of lattice sites.
        n_mixture: Each layer is a convex combination of `n_mixture` transformations.
        net_shape: Hidden layer widths for the fully-connected neural networks.
        net_activation: Activation function to apply after each linear transformation.
        weighted: Whether or not the convex combinations are weighted mixtures.
        min_weight: Minimum weight for weighted mixtures.
        ramp_pow: Integer power in the ramp function used to construct sigmoid transformations.
    """
    def __init__(
        self,
        lattice_size: int,
        n_mixture: int,
        net_shape: list[int],
        net_activation: str = "Tanh",
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp_pow: int = 2,
    ):
        super().__init__()
        transforms = [
            build_sigmoid_module(
                net_shape=net_shape,
                net_activation=net_activation,
                n_mixture=n_mixture,
                weighted=weighted,
                min_weight=min_weight,
                ramp_pow=ramp_pow,
            )
            for _ in range(lattice_size - 1)
        ]
        self.register_module("transforms", nn.ModuleList(transforms))

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        assert φ.shape[1] == len(self.transforms) + 1
        return self._forward_v4(φ)

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

            context = as_vector(
                U.transpose(1, 2)
            )  # .sum(dim=-1, keepdim=True))

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


class CouplingFlow(Flow):
    def __init__(
        self,
        n_blocks: int,
        n_mixture: int,
        net_shape: list[int],
        net_activation: str = "Tanh",
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp_pow: int = 2,
    ):
        super().__init__()
        transforms = [
            build_sigmoid_module(
                net_shape=net_shape,
                net_activation=net_activation,
                n_mixture=n_mixture,
                weighted=weighted,
                min_weight=min_weight,
                ramp_pow=ramp_pow,
            )
            for _ in range(4 * n_blocks)
        ]
        self.register_module("transforms", nn.ModuleList(transforms))

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        return self._forward_v1(φ)

    def _forward_v1(self, φ):
        assert φ.shape[1] % 2 == 0
        # NOTE: if we make angles relative to φ0 and don't also mask φ0 such that
        # it never gets transformed, we break the triangular Jacobian structure!!
        φ0, _ = φ.tensor_split([1], dim=1)
        φ = mod_2pi(φ - φ0)

        checker = φ.new_zeros(φ.shape[1]).bool()
        checker[::2] = True
        m1, m2 = checker, ~checker
        m1[0] = False  # don't transform φ0 or things break!

        ldj_total = 0.0
        for transform, mask in zip(self.transforms, cycle((m1, m2))):
            φ_t = φ.clone()

            neighbours = torch.cat([φ.roll(-1, 1), φ.roll(+1, 1)], dim=-1)
            context = as_vector(neighbours[:, mask])
            f = transform(context)
            φ_t[:, mask], ldj = f(φ[:, mask])

            ldj_total += ldj
            φ = φ_t

        φ = mod_2pi(φ + φ0)

        return φ, ldj_total

    def _forward_v2(self, φ):
        """
        The masking pattern is

        o --- o ~~~ x --- o --- o
           F    A     P     F

        A, P, F = active, passive frozen link variables
        U_x = φ_x - φ_{x-1}
        """
        assert φ.shape[1] % 3 == 0

        mask = φ.new_zeros(φ.shape[1]).bool()
        mask[::3] = True

        ldj_total = 0.0
        for transform in self.transforms:
            U = mod_2pi(φ - φ.roll(+1, 1))

            # assert torch.allclose(U.roll(-1, 1), mod_2pi(φ.roll(-1, 1) - φ))
            # assert torch.allclose(U.roll(+1, 1), mod_2pi(φ.roll(+1, 1) - φ.roll(+2, 1)))
            # assert torch.allclose(U.roll(-2, 1), mod_2pi(φ.roll(-2, 1) - φ.roll(-1, 1)))

            # Stack the two frozen links equidistant from the active spin.
            # I.e. if φ_x is the active spin and U_x is the active link, then stack
            # U_{x-1} = (φ_{x-1} - φ_{x-2}) and U_{x+2} = (φ_{x+2} - φ_{x+1}) on x
            context = as_vector(
                torch.cat([U.roll(+1, 1), U.roll(-2, 1)], dim=-1)[:, mask]
            )
            f = transform(context)

            U_t = U.clone()
            U_t[:, mask], ldj = f(U[:, mask])

            # Change variables back to spins
            φ = torch.where(
                mask.view(1, -1, 1), mod_2pi(U_t + φ.roll(+1, 1)), φ
            )

            ldj_total += ldj

            mask = mask.roll(-1)

        return φ, ldj_total
