from math import pi as π
from functools import partial

import torch
from torch.func import vmap, jacrev
import torch.nn.functional as F

from nflows_xy.transforms.module import build_sigmoid_module
from nflows_xy.transforms.sigmoid import (
    exponential_ramp,
    sigmoid_,
    affine_,
    build_sigmoid_transform,
)

torch.set_default_dtype(torch.double)

ramp = partial(exponential_ramp, power=2, eps=1e-6)
sigmoid = sigmoid_(ramp)
affine_sigmoid = affine_(sigmoid)

N = 10000
L = 12


def make_func(func):
    def _func(x, params):
        y, dydx = func(x.view(1, -1, 1), params.unsqueeze(0))
        return y.view(-1), dydx.view(-1)

    vmapped_jacrev = vmap(jacrev(_func, argnums=0, has_aux=True))

    def _func2(x, params):
        auto, man = vmapped_jacrev(x, params)
        return auto.diagonal(dim1=-2, dim2=-1), man

    return _func2

def make_func_v2(cls):
    def _func(x, params):
        func = cls(params.unsqueeze(0))
        y, dydx = func(x.view(1, -1, 1))
        return y.view(-1), dydx.view(-1)

    vmapped_jacrev = vmap(jacrev(_func, argnums=0, has_aux=True))

    def _func2(x, params):
        auto, man = vmapped_jacrev(x, params)
        return auto.diagonal(dim1=-2, dim2=-1), man

    return _func2


def test_grad_agrees_with_autograd():
    print("")
    x = torch.empty(N, L).uniform_(0, 1)
    x.requires_grad_(True)

    a = F.softplus(x.new_empty(N, L, 1).normal_()) + 2
    α = F.softplus(x.new_empty(N, L, 1).normal_()) - 2
    #α = x.new_empty(N, L, 1).normal_().sub(10).negative().exp()
    β = torch.sigmoid(x.new_empty(N, L, 1).normal_())
    c = x.new_ones(N, L, 1)

    f = make_func(ramp)
    auto, man = f(x, a)
    print(man.mean().item(), man.std().item(), man.abs().max().item())
    assert torch.allclose(man, auto)

    f = make_func(sigmoid)
    auto, man = f(x, a)
    print(man.mean().item(), man.std().item(), man.abs().max().item())
    assert torch.allclose(man, auto)

    f = make_func(affine_sigmoid)
    auto, man = f(x, torch.cat((a, α, β), dim=-1))
    print(man.mean().item(), man.std().item(), man.abs().max().item())
    assert torch.allclose(man, auto)

    #f = make_func_v2(build_sigmoid_transform(n_mixture=1, weighted=False))
    #auto, man = f(x * 2 * π, torch.cat((a, α, β, c), dim=-1))
    #print(man.mean().item(), man.std().item(), man.abs().max().item())
    #assert torch.allclose(man, auto)
