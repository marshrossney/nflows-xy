"""
Credit to https://arxiv.org/pdf/2110.00351.pdf
and https://github.com/noegroup/bgflow for this transformation
"""
from functools import partial
from math import pi as π
from typing import Any, Callable, TypeAlias

import torch
import torch.nn.functional as F

from .wrappers import (
    rescale_to_interval_,
    mixture_,
    mix_with_identity_,
)
from .utils import normalise_weights

Tensor: TypeAlias = torch.Tensor
TransformFunc: TypeAlias = Callable[[Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]]
Transform: TypeAlias = Callable[Tensor, tuple[Tensor, Tensor]]


def exponential_ramp(
    x: Tensor, params: Tensor, *, power: int = 2, eps: float = 1e-9
) -> tuple[Tensor, Tensor]:
    assert isinstance(power, int) and power > 0
    a, b, ε = params, power, eps
    x_masked = torch.where(x > ε, x, torch.full_like(x, ε))
    exp_factor = -a * x_masked.pow(-b)
    ρ = torch.where(
        x > ε,
        torch.exp(exp_factor) / torch.exp(-a),
        torch.zeros_like(x),
    )
    dρdx = (-b / x_masked) * exp_factor * ρ
    return ρ, dρdx


def sigmoid_(ramp: TransformFunc) -> TransformFunc:
    def sigmoid(x: Tensor, params: Tensor, **kwargs):
        ρ_x, dρdx_x = ramp(x, params, **kwargs)
        ρ_1mx, dρdx_1mx = ramp(1 - x, params, **kwargs)

        σ = ρ_x / (ρ_x + ρ_1mx)

        dσdx = (ρ_1mx * dρdx_x + ρ_x * dρdx_1mx) / (ρ_x + ρ_1mx) ** 2

        return σ, dσdx

    return sigmoid


def affine_(
    sigmoid: TransformFunc,
) -> TransformFunc:
    def affine(x: Tensor, params: Tensor, **kwargs):
        params, α, β = params.tensor_split([-2, -1], dim=-1)
        σ, dσdx = sigmoid((x - β) * α + 0.5, params, **kwargs)
        return σ, α * dσdx

    return affine


def build_sigmoid_transform(
    n_mixture: int,
    weighted: bool = True,
    ramp_pow: int = 2,
    min_weight: float = 1e-2,
) -> Transform:
    weighted = weighted if n_mixture > 1 else False

    ramp = partial(exponential_ramp, power=ramp_pow)

    transform = mixture_(
        rescale_to_interval_(
            mix_with_identity_(affine_(sigmoid_(ramp))),
            lower_bound=0.0,
            upper_bound=2 * π,
        ),
        weighted=weighted,
        mixture_dim=-2,
    )

    funcs = [
        lambda x: F.softplus(x) + 1e-3,  # exponential ramp 'a'
        lambda x: x.negative().exp(),  # affine 'α'
        torch.sigmoid,  # affine 'β'
        torch.sigmoid,  # weight wrt identity transform
    ]
    if weighted:
        funcs.append(partial(normalise_weights, dim=-2, min=min_weight))

    def handle_params(params: Tensor) -> Tensor:
        params = params.unflatten(-1, (n_mixture, -1)).split(1, dim=-1)
        params = torch.cat(
            [func(param) for func, param in zip(funcs, params, strict=True)],
            dim=-1,
        )
        return params

    class SigmoidTransform:
        n_params = (4 + int(weighted)) * n_mixture

        def __init__(self, params: Tensor):
            self.params = handle_params(params)

        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return transform(x, self.params)

    return SigmoidTransform
