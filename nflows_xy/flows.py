from abc import ABC, abstractmethod
from itertools import cycle
from math import isclose, log, pi as π

import torch
import torch.nn as nn

from nflows_xy.transforms import build_sigmoid_module
from nflows_xy.utils import mod_2pi, as_vector, mean_angle

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

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        φ0, *z = z.split(1, dim=1)

        φ = φ0
        ldj_total = 0.0
        for z_x, transform in zip(z, self.transforms, strict=True):
            # Construct conditional transformation using context Σ_{i=0}^{x-1} U_i
            ΣU = (
                (φ[:, 1:] - φ[:, :-1])
                .transpose(1, 2)
                .sum(dim=-1, keepdim=True)
            )
            context = as_vector(ΣU)
            f = transform(context)

            # Transform V_x -> U_x = f(V_x | Σ_{i=0}^{x-1} U_i)
            _, φ_x_minus_1 = φ.tensor_split([-1], dim=1)
            V_x = mod_2pi(z_x - φ_x_minus_1)
            U_x, ldj_x = f(V_x)
            φ_x = mod_2pi(U_x + φ_x_minus_1)

            φ = torch.cat([φ, φ_x], dim=1)
            ldj_total += ldj_x

        return φ, ldj_total


class HierarchicalFlow(Flow):
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
        # Check lattice size is a power of two
        assert lattice_size & (lattice_size - 1) == 0
        n_layers = log(lattice_size, 2)
        assert isclose(n_layers, int(n_layers))
        transforms = [
            build_sigmoid_module(
                net_shape=net_shape,
                net_activation=net_activation,
                n_mixture=n_mixture,
                weighted=weighted,
                min_weight=min_weight,
                ramp_pow=ramp_pow,
            )
            for _ in range(int(n_layers))
        ]
        self.register_module("transforms", nn.ModuleList(transforms))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        φ, z = z.tensor_split([1], dim=1)
        ldj_total = 0.0

        for transform in self.transforms:
            # Difference between neighbouring angles
            context = as_vector(φ - φ.roll(-1, 1))
            f = transform(context)

            # Double the number of degrees of freedom
            z_x, z = z.tensor_split([φ.shape[1]], dim=1)

            # Transform uniform variates
            φ_x, ldj = f(z_x)
            ldj_total += ldj

            # Rotate to average angle of neighbouring spins
            α = mean_angle(φ, φ.roll(-1, 1))
            φ_x = mod_2pi(φ_x + α)

            # Interleave the new φ between the existing ones
            L = φ.shape[1]
            φ = torch.cat([φ, φ_x], dim=-1).view(-1, 2 * L, 1)

        assert z.numel() == 0

        return φ, ldj_total


class SpinCouplingFlow(Flow):
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
            for _ in range(2 * n_blocks)
        ]
        self.register_module("transforms", nn.ModuleList(transforms))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        assert z.shape[1] % 2 == 0

        checker = z.new_zeros(z.shape[1]).bool()
        checker[::2] = True
        m1, m2 = checker, ~checker
        m1[0] = False  # don't transform φ0 or things break!

        # Make all angles relative to φ0
        # NOTE: take φ0 = π since transformations maximally flexible here
        φ0, _ = z.tensor_split([1], dim=1)
        φ = mod_2pi(z - φ0 + π)

        ldj_total = 0.0
        for transform, mask in zip(self.transforms, cycle((m1, m2))):
            z = φ

            # Just take the nearest neighbours as context
            neighbours = torch.cat([z.roll(-1, 1), z.roll(+1, 1)], dim=-1)
            context = as_vector(neighbours[:, mask])
            f = transform(context)

            φ = z.clone()
            φ[:, mask], ldj = f(z[:, mask])

            ldj_total += ldj

            φ = mod_2pi(φ + π / 2)

        φ = mod_2pi(φ + φ0 - π)

        return φ, ldj_total


class LinkCouplingFlow(Flow):
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
            for _ in range(3 * n_blocks)
        ]
        self.register_module("transforms", nn.ModuleList(transforms))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        The masking pattern is

        o --- o ~~~ x --- o --- o
           F     A     P     F

        A, P, F = active, passive frozen link variables
        U_x = φ_x - φ_{x-1}
        """
        _, L, _ = z.shape
        assert L % 3 == 0

        mask = torch.tensor(
            [True, False, False], dtype=torch.bool, device=z.device
        ).repeat(L // 3)

        ldj_total = 0.0
        for transform in self.transforms:
            V = mod_2pi(z - z.roll(+1, 1))

            # assert torch.allclose(U.roll(-1, 1), mod_2pi(φ.roll(-1, 1) - φ))
            # assert torch.allclose(U.roll(+1, 1), mod_2pi(φ.roll(+1, 1) - φ.roll(+2, 1)))
            # assert torch.allclose(U.roll(-2, 1), mod_2pi(φ.roll(-2, 1) - φ.roll(-1, 1)))

            # Stack the two frozen links equidistant from the active spin.
            # I.e. if φ_x is the active spin and U_x is the active link, then stack
            # U_{x-1} = (φ_{x-1} - φ_{x-2}) and U_{x+2} = (φ_{x+2} - φ_{x+1}) on x
            context = as_vector(
                torch.cat([V.roll(+1, 1), V.roll(-2, 1)], dim=-1)[:, mask]
            )
            f = transform(context)

            U = V.clone()
            U[:, mask], ldj = f(V[:, mask])

            # Change variables back to spins
            φ = torch.where(mask.view(1, -1, 1), mod_2pi(U + z.roll(+1, 1)), z)

            ldj_total += ldj
            z = φ

            # Roll mask to select next set of spins
            mask = mask.roll(-1)

        return φ, ldj_total
