from abc import ABC, abstractmethod
from math import pi as π

import torch

from nflows_xy.utils import mod_2pi

Tensor = torch.Tensor


def top_charge(φ: Tensor) -> Tensor:
    assert φ.shape[-1] == 1
    U = φ - φ.roll(+1, -2)
    q = (mod_2pi(U + π) - π) / (2 * π)
    return q.sum(dim=-2)


class Action(ABC):
    def __init__(self, beta: float, lattice_size: int, lattice_dim: int):
        self.beta = beta
        self.lattice_size = lattice_size
        self.lattice_dim = lattice_dim

    @property
    def lattice_shape(self) -> tuple[int]:
        return tuple(self.lattice_size for _ in range(self.lattice_dim))

    @abstractmethod
    def __call__(self, φ: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad(self, φ: Tensor) -> Tensor:
        ...


class Action1d(Action):
    def __init__(self, beta: float, lattice_size: int):
        super().__init__(beta, lattice_size, 1)

    def __call__(self, φ: Tensor) -> Tensor:
        return -self.beta * torch.cos(φ - φ.roll(+1, -2)).sum(dim=-2)

    def grad(self, φ: Tensor) -> Tensor:
        return -self.beta * (
            -torch.sin(φ - φ.roll(+1, -2)) + torch.sin(φ.roll(-1, -2) - φ)
        )


class Action2d(Action):
    def __init__(self, beta: float, lattice_size: int):
        super().__init__(beta, lattice_size, 2)

    def __call__(self, φ: Tensor) -> Tensor:
        return -self.beta * (
            torch.cos(φ - φ.roll(+1, -3)).sum(dim=(-3, -2))
            + torch.cos(φ - φ.roll(+1, -2)).sum(dim=(-3, -2))
        )

    def grad(self, φ: Tensor) -> Tensor:
        return -self.beta * (
            -torch.sin(φ - φ.roll(+1, -3))
            + torch.sin(φ.roll(-1, -3) - φ)
            - torch.sin(φ - φ.roll(+1, -2))
            + torch.sin(φ.roll(-1, -2) - φ)
        )


def action(
    beta: float,
    lattice_size: int,
    lattice_dim: int,
) -> Action:
    """Specify an XY action that can be used as a training or HMC target.

    Args:
        beta: Inverse temperature parameter, (β)
        lattice_size: Number of lattice sites in each dimension
        lattice_dim: Number of lattice dimensions (1 or 2)
    """
    if lattice_dim == 1:
        return Action1d(beta, lattice_size)
    elif lattice_dim == 2:
        return Action2d(beta, lattice_size)
    else:
        raise ValueError("Only one or two dimensions supported")
