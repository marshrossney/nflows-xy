from functools import lru_cache
from math import pi as π
from typing import Any, Callable, TypeAlias

import torch

from nflows_xy.utils import mod_2pi

Tensor: TypeAlias = torch.Tensor
TransformFunc: TypeAlias = Callable[
    [Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]
]
Transform: TypeAlias = Callable[Tensor, tuple[Tensor, Tensor]]


@lru_cache
def mixture_(
    transform: TransformFunc,
    weighted: bool = True,
    mixture_dim: int = -2,
) -> TransformFunc:
    d = mixture_dim
    vmapped_transform = torch.vmap(transform, (None, d), (d, d))

    if weighted:

        def mixture_transform(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            params, weights = params.tensor_split([-1], dim=-1)
            # NOTE: vmap/allclose support github.com/pytorch/functorch/issues/275
            # assert torch.allclose(weights.sum(dim=d), torch.ones(1))
            y, dydx = vmapped_transform(x, params, **kwargs)
            assert y.shape == weights.shape
            return (weights * y).sum(dim=d), (weights * dydx).sum(dim=d)

    else:

        def mixture_transform(
            x: Tensor, params: Tensor, **kwargs
        ) -> tuple[Tensor, Tensor]:
            y, dydx = vmapped_transform(x, params, **kwargs)
            return y.mean(dim=d), dydx.mean(dim=d)

    return mixture_transform


def mix_with_identity_(transform: TransformFunc) -> TransformFunc:
    def identity_mixture(
        x: Tensor, params: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor]:
        params, c = params.tensor_split([-1], dim=-1)
        # assert torch.all((c >= 0) and (c <= 1))

        y, dydx = transform(x, params, **kwargs)

        y = c * y + (1 - c) * x
        dydx = c * dydx + (1 - c)

        return y, dydx

    return identity_mixture


def rescale_to_interval_(
    transform: TransformFunc,
    lower_bound: float,
    upper_bound: float,
) -> TransformFunc:
    lo, up = lower_bound, upper_bound
    assert lo < up

    def rescaled_transform(x: Tensor, *args, **kwargs):
        y, dydx = transform((x - lo) / (up - lo), *args, **kwargs)
        return y * (up - lo) + lo, dydx

    return rescaled_transform


def pi_rotation_(transform):
    def wrapped_transform(θ: Tensor, *args, **kwargs):
        θ, grad_or_ldj = transform(mod_2pi(θ + π))
        return mod_2pi(θ - π), grad_or_ldj

    return wrapped_transform


def mask_outside_interval_(
    transform,
    *,
    lower_bound: float,
    upper_bound: float,
    tol: float,
):
    assert lower_bound < upper_bound

    def wrapped_transform(x: Tensor, *args, **kwargs):
        inside_bounds = (x > lower_bound + tol) & (x < upper_bound - tol)

        y, grad = transform(
            x.clamp(lower_bound + tol, upper_bound - tol), *args, **kwargs
        )

        y = torch.where(inside_bounds, y, x)
        # NOTE: require grad, not ldj!
        grad = torch.where(inside_bounds, grad, torch.ones_like(grad))

        return y, grad

    return wrapped_transform


def sum_log_gradient_(transform):
    def wrapped_transform(x: Tensor, *args, **kwargs):
        y, grad = transform(x, *args, **kwargs)
        sum_log_grad = grad.log().flatten(start_dim=1).sum(dim=1, keepdim=True)
        return y, sum_log_grad

    return wrapped_transform


def make_hook(wrapper):
    def hook(module, inputs, outputs):
        return wrapper(outputs)

    return hook
