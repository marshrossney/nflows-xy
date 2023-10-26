from functools import partial
from math import pi as π
from typing import Any, Callable, TypeAlias

import torch

from nflows_xy.utils import mod_2pi, as_angle, as_vector, dot_keepdim
from .wrappers import mixture_
from .utils import normalise_weights

Tensor: TypeAlias = torch.Tensor
TransformFunc: TypeAlias = Callable[
    [Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]
]
Transform: TypeAlias = Callable[Tensor, tuple[Tensor, Tensor]]

def mobius(x: Tensor, ω: Tensor) -> tuple[Tensor, Tensor]:
    x = as_vector(x)
    dydx = (1 - dot_keepdim(ω, ω)) / dot_keepdim(x - ω, x - ω)
    y = dydx * (x - ω) - ω
    y = as_angle(y)
    return y, dydx

def mobius_for_mixture(x: Tensor, ω: Tensor) -> tuple[Tensor, Tensor]:
    x_x0 = torch.stack([as_vector(x), as_vector(torch.zeros_like(x))])

    dydx = (1 - dot_keepdim(ω, ω)) / dot_keepdim(x_x0 - ω, x_x0 - ω)
    y_y0 = dydx * (x_x0 - ω) - ω

    y, y0 = as_angle(y_y0)
    y = mod_2pi(y - y0)

    dydx, _ = dydx

    return y, dydx


def to_unit_disk(ω: Tensor, ε: float) -> Tensor:
    assert ω.shape[-1] == 2
    ω = torch.tanh(ω) * (1 - ε)
    ω1, ω2 = ω.split(1, dim=-1)
    return torch.cat([ω1, ω2 * (1 - ω1**2).sqrt()], dim=-1)

"""
def build_mobius_transform(
    n_mixture: int,
    weighted: bool = True,
    min_weight: float = 1e-2,
    bounds_tol: float = 1e-2,
) -> Transform:
    weighted = weighted if n_mixture > 1 else False

    if n_mixture > 1:
        forward_fn = vmap(
            make_mixture(_forward_transform, weighted=weighted, mixture_dim=-2)
        )
        inverse_fn = invert_bisect(
            forward_fn,
            0,
            2 * π,
            tol=invert_bisect_tol,
            max_iter=invert_bisect_max_iter,
        )
    else:
        forward_fn = vmap(_forward_transform)
        inverse_fn = vmap(_inverse_transform)

    funcs = [partial(to_unit_disk, ε=bounds_tol)]
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).tensor_split(
            [2], dim=-1
        )
        # NOTE zip with strict=False so empty tensor dropped when unweighted
        return torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=False)],
            dim=-1,
        ).squeeze(dim=-2)

    class MobiusTransform:
        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return forward_fn(x, self.params)

        def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
            return inverse_fn(y, self.params)

    return MobiusTransform

"""
