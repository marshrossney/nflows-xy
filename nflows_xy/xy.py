from abc import ABC, abstractmethod
import logging
from math import pi as π

import scipy
import torch

from nflows_xy.utils import mod_2pi, log_cosh

Tensor = torch.Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def top_charge(φ: Tensor) -> Tensor:
    assert φ.shape[-1] == 1
    U = φ - φ.roll(+1, -2)
    q = (mod_2pi(U + π) - π) / (2 * π)
    return q.sum(dim=-2)


def spin_correlation(φ: Tensor) -> Tensor:
    """
    G(δx) = E[σ(x) . σ(x+δx)] - E[σ(x)]E[σ(x+δx)]
          = E[cos(φ(x) - φ(x+δx))] - (0)(0)
    """
    assert φ.shape[-1] == 1
    L = φ.shape[-2]
    G = []
    for δx in range(L):
        # Take volume average as well as ensemble average
        G.append(torch.cos(φ - φ.roll(δx, -2)).mean(dim=(-3, -2)))
    return torch.cat(G, dim=-1)


def fit_spin_correlation(G: Tensor) -> Tensor:
    """
    Quick and dirty log cosh fit for the spin correlation function.

    Caveats:
    - Errors not treated properly: should probably bootstrap this or at least
      pass sample std dev to optimize.curve_fit
    - Fit includes all points except δx = 0, which is not quite right
    """
    _, L = G.shape
    log_G = G.mean(0).log()

    def func_to_fit(x: Tensor, ξ: float, c: float):
        # see note on scipy.optimizer.curve_fit docs
        assert x.dtype == torch.float64
        log_G_fit = log_cosh((x - L / 2) / ξ) + c
        return log_G_fit.numpy()

    ydata = log_G[1:]
    mask = ydata.isfinite()
    ydata = ydata[mask]
    xdata = torch.arange(1, L)[mask]

    popt, pcov = scipy.optimize.curve_fit(
        func_to_fit,
        xdata=xdata.to(torch.float64),
        ydata=ydata.to(torch.float64),
        p0=(L / 4, 0.0),
        bounds=((0.0, -float("inf")), (float("inf"), float("inf"))),
    )
    return torch.from_numpy(popt), torch.from_numpy(pcov)


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
    else:
        raise ValueError("Only one dimension supported so far")
