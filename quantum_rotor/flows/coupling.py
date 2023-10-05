from itertools import cycle, islice, repeat

import torch
import torch.nn as nn

from quantum_rotor.transforms import build_sigmoid_module
from quantum_rotor.utils import mod_2pi, as_vector

Tensor = torch.Tensor


class CouplingFlow(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        assert len(transforms) % 4 == 0
        self.transforms = nn.ModuleList(transforms)

    def _forward_v1(self, φ):
        assert φ.shape[1] % 2 == 0
        # NOTE: if we make angles relative to φ0 and don't also mask φ0 such that
        # it never gets transformed, we break the triangular Jacobian structure!!
        φ0, _ = φ.tensor_split([1], dim=1)
        φ = mod_2pi(φ - φ0)

        checker = φ.new_zeros(φ.shape[1]).bool()
        checker[::2] = True
        m1, m2 = checker, ~checker
        m1[0] = False  # don't transform φ0 or things break!

        ldj_total = 0.0
        for transform, mask in zip(self.transforms, cycle((m1, m2))):
            φ_t = φ.clone()

            neighbours = torch.cat([φ.roll(-1, 1), φ.roll(+1, 1)], dim=-1)
            context = as_vector(neighbours[:, mask])
            f = transform(context)
            φ_t[:, mask], ldj = f(φ[:, mask])

            ldj_total += ldj
            φ = φ_t

        φ = mod_2pi(φ + φ0)

        return φ, ldj_total

    def _forward_v2(self, φ):
        """
        The masking pattern is

        o --- o ~~~ x --- o --- o
           F1    A     P     F2

        A, P, F = active, passive frozen link variables
        U_x = φ_x - φ_{x-1}
        """
        assert φ.shape[1] % 3 == 0

        mask = φ.new_zeros(φ.shape[1]).bool()
        mask[::3] = True

        ldj_total = 0.0
        for transform in self.transforms:
            U = mod_2pi(φ - φ.roll(+1, 1))

            # assert torch.allclose(U.roll(-1, 1), mod_2pi(φ.roll(-1, 1) - φ))
            # assert torch.allclose(U.roll(+1, 1), mod_2pi(φ.roll(+1, 1) - φ.roll(+2, 1)))
            # assert torch.allclose(U.roll(-2, 1), mod_2pi(φ.roll(-2, 1) - φ.roll(-1, 1)))

            # Stack the two frozen links equidistant from the active spin.
            # I.e. if φ_x is the active spin and U_x is the active link, then stack
            # U_{x-1} = (φ_{x-1} - φ_{x-2}) and U_{x+2} = (φ_{x+2} - φ_{x+1}) on x
            context = as_vector(
                torch.cat([U.roll(+1, 1), U.roll(-2, 1)], dim=-1)[:, mask]
            )
            f = transform(context)

            U_t = U.clone()
            U_t[:, mask], ldj = f(U[:, mask])

            # Change variables back to spins
            φ = torch.where(mask.view(1, -1, 1), mod_2pi(U_t + φ.roll(+1, 1)), φ)

            ldj_total += ldj

            mask = mask.roll(-1)

        return φ, ldj_total

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        return self._forward_v1(φ)


class SigmoidCouplingFlow(CouplingFlow):
    def __init__(
        self,
        n_blocks: int,
        n_lattice: int,
        n_mixture: int,
        net_shape: list[int],
        net_activation: str = "Tanh",
        weighted: bool = True,
        min_weight: float = 1e-2,
        ramp_pow: int = 2,
    ):
        transforms = [
            build_sigmoid_module(
                net_shape=net_shape,
                net_activation=net_activation,
                n_mixture=n_mixture,
                weighted=weighted,
                min_weight=min_weight,
                ramp_pow=ramp_pow,
            )
            for _ in range(4 * n_blocks)
        ]
        super().__init__(transforms)
