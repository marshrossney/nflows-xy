from enum import Enum
from itertools import chain
import warnings

import torch.nn as nn

warnings.filterwarnings("ignore", message="Lazy")


class StrEnum(str, Enum):
    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return str(self) == str(other)


class Activation(StrEnum):
    Identity = 0
    ELU = 1
    Hardshrink = 2
    Hardsigmoid = 3
    Hardtanh = 4
    Hardswish = 5
    LeakyReLU = 6
    LogSigmoid = 7
    PReLU = 8
    ReLU = 9
    ReLU6 = 10
    RReLU = 11
    SELU = 12
    CELU = 13
    GELU = 14
    Sigmoid = 15
    SiLU = 16
    Mish = 17
    Softplus = 18
    Softshrink = 19
    Softsign = 20
    Tanh = 21
    Tanhshrink = 22
    Threshold = 23
    GLU = 24


def permute_io(net: nn.Module) -> nn.Module:
    def permute_input(module, inputs):
        (input,) = inputs
        return input.permute(0, 2, 1)

    def permute_output(module, inputs, output):
        return output.permute(0, 2, 1)

    # NOTE: prepend required so that permute occurs before shape inference
    # when using LazyModule
    net.register_forward_pre_hook(permute_input, prepend=True)
    net.register_forward_hook(permute_output)

    return net


def build_cnn(
    channels: list[int],
    kernel_radius: int | list[int],
    activation: Activation | None,
) -> nn.Sequential:
    activation = activation or "Identity"

    if isinstance(kernel_radius, int):
        kernel_radius = [kernel_radius for _ in channels]

    conv_layers = [
        nn.LazyConv1d(
            n, kernel_size=(2 * r + 1), padding=r, padding_mode="circular"
        )
        for n, r in zip(channels, kernel_radius, strict=True)
    ]
    activations = [getattr(nn, str(activation))() for _ in conv_layers]
    layers = list(chain(*zip(conv_layers, activations)))
    net = nn.Sequential(*layers)

    return permute_io(net)


def build_fnn(
    widths: list[int],
    activation: Activation | None,
) -> nn.Sequential:
    activation = activation or "Identity"
    linear_layers = [nn.LazyLinear(n) for n in widths]
    activations = [getattr(nn, str(activation))() for _ in linear_layers]
    layers = list(chain(*zip(linear_layers, activations)))
    net = nn.Sequential(*layers)
    return net
