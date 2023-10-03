from math import pi as π

import torch
from torch import einsum

Tensor = torch.Tensor


def mod_2pi(φ: Tensor) -> Tensor:
    return torch.remainder(φ, 2 * π)


def as_angle(x: Tensor) -> Tensor:
    return mod_2pi(torch.atan2(*list(reversed(x.split(1, dim=-1)))))


def as_vector(φ: Tensor) -> Tensor:
    return torch.cat([φ.cos(), φ.sin()], dim=-1)


def dot(u: Tensor, v: Tensor) -> Tensor:
    return einsum("...i,...i->...", u, v)


def dot_keepdim(u: Tensor, v: Tensor) -> Tensor:
    return dot(u, v).unsqueeze(-1)
