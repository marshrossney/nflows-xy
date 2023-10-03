import torch
import torch.nn as nn


from quantum_rotor.transforms import build_sigmoid_module

Tensor = torch.Tensor


class AutoregressiveFlow(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, φ: Tensor) -> tuple[Tensor, Tensor]:
        assert φ.shape[1] == len(self.transforms) + 1

        ldj = 0.0
        θ, φ = φ.tensor_split([1], dim=1)
        for φ_t, f_t in zip(φ.split(1, dim=1), self.transforms):
            θ_t, ldj_t = f_t(θ.transpose(1, 2))(φ_t)

            θ = torch.cat([θ, θ_t], dim=1)
            ldj += ldj_t

        return θ, ldj


class AutoregressiveSigmoidFlow(AutoregressiveFlow):
    def __init__(
        self,
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
            for _ in range(n_lattice - 1)
        ]
        super().__init__(transforms)
