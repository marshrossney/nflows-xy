from typing import Callable

import torch
import torch.nn as nn

from quantum_rotor.nn import build_fnn

from .sigmoid import build_sigmoid_transform
from .wrappers import sum_log_gradient_, make_hook

Tensor = torch.Tensor


class UnivariateTransformModule(nn.Module):
    def __init__(
        self,
        transform_factory,
        context_fn: Callable[Tensor, Tensor] | None = None,
        wrappers: list[Callable] = [],
    ):
        super().__init__()
        self.transform_factory = transform_factory
        self.context_fn = (
            context_fn if context_fn is not None else nn.Identity()
        )

        self._hooks = {
            wrapper.__name__: self.register_forward_hook(make_hook(wrapper))
            for wrapper in wrappers
        }

    def forward(self, context: Tensor):
        params = self.context_fn(context)
        transform = self.transform_factory(params)
        return transform


def _build_univariate_module(
    Transform,
    net_shape: list[int],
    net_activation: str,
    wrappers: list,
):
    if len(net_shape) > 0:
        net = build_fnn(net_shape, net_activation)
        net.append(nn.LazyLinear(Transform.n_params))
    else:
        net = nn.LazyLinear(Transform.n_params)

    return UnivariateTransformModule(Transform, net, wrappers)


def build_sigmoid_module(
    net_shape: list[int],
    net_activation: str,
    n_mixture: int,
    weighted: bool,
    min_weight: float,
    ramp_pow: int,
) -> UnivariateTransformModule:
    Transform = build_sigmoid_transform(
        n_mixture=n_mixture,
        weighted=weighted,
        min_weight=min_weight,
        ramp_pow=ramp_pow,
    )
    wrappers = [sum_log_gradient_]
    return _build_univariate_module(
        Transform, net_shape, net_activation, wrappers
    )
