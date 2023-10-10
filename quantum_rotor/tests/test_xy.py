from math import pi as π

import torch

from quantum_rotor.xy import action


def test_action_grad_agrees_with_autograd():
    φ = torch.empty(2, 12, 1).uniform_(0, 2 * π)
    φ.requires_grad_(True)

    for dim in (1, 2):
        S = action(2.0, 12, 1)
        φ.grad = None
        (grad,) = torch.autograd.grad(
            outputs=S(φ),
            inputs=φ,
            grad_outputs=torch.ones(φ.shape[0], 1),
        )
        assert torch.allclose(grad, S.grad(φ))
