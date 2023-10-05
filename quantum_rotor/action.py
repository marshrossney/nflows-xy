from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# from torch.special import i0e

Tensor = torch.Tensor


class Action(ABC):
    @abstractmethod
    def __call__(self, inputs: Tensor) -> Tensor:
        ...

    @abstractmethod
    def grad(self, inputs: Tensor) -> Tensor:
        ...


class ActionPBC(Action):
    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, φ: Tensor) -> Tensor:
        return -self.beta * torch.cos(φ - φ.roll(+1, 1)).sum(dim=1)

    def grad(self, φ: Tensor) -> Tensor:
        return -self.beta * (
            -torch.sin(φ - φ.roll(+1, 1)) + torch.sin(φ.roll(-1, 1) - φ)
        )


class ActionOBC(Action):
    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, φ: Tensor) -> Tensor:
        return -self.beta * torch.cos(φ[:, 1:] - φ[:, :-1]).sum(dim=1)

    def grad(self, φ: Tensor) -> Tensor:
        g = -torch.sin(φ[:, 1:] - φ[:, :-1])
        out = torch.zeros_like(φ)
        out[:, 1:] += g
        out[:, :-1] -= g
        return -self.beta * out


class PullbackAction(nn.Module, Action):
    def __init__(self, action: Action, flow: nn.Module):
        super().__init__()
        self.action = action
        self.register_module("flow", flow)

    @torch.no_grad()
    def __call__(self, u: Tensor) -> Tensor:
        φ, ldj = self.flow(u)
        return self.action(φ) - ldj

    @torch.enable_grad()
    def grad(self, u: Tensor) -> Tensor:
        u.requires_grad_(True)
        φ, ldj = self.flow(u)

        pullback = self.action(φ) - ldj

        (gradient,) = torch.autograd.grad(
            outputs=pullback,
            inputs=u,
            grad_outputs=torch.ones_like(pullback),
        )

        return gradient


def _test():
    from math import pi as π

    φ = torch.empty(2, 12, 1).uniform_(0, 2 * π)
    φ.requires_grad_(True)

    for action in (ActionPBC(2.0), ActionOBC(2.0)):
        φ.grad = None
        (grad,) = torch.autograd.grad(
            outputs=action(φ),
            inputs=φ,
            grad_outputs=torch.ones(φ.shape[0], 1),
        )
        assert torch.allclose(grad, action.grad(φ)), f"problem with {action}"


if __name__ == "__main__":
    _test()
