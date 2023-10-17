from math import log, pi as π

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


def log_cosh(x: Tensor) -> Tensor:
    # Numerically stable implementation of log(cosh(x))
    return abs(x) + torch.log1p(torch.exp(-2 * abs(x))) - log(2)


def make_banner(txt: str, size: int = 79) -> str:
    assert len(txt) < size
    rule = "=" * size
    pad = (size - len(txt)) // 2
    return rule + "\n" + txt.rjust(pad) + "\n" + rule
