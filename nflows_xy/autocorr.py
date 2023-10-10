from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _compute_autocorrelation(X: Tensor) -> Tensor:
    X = X.squeeze(-1)
    assert X.dim() == 1

    X = X - X.mean()

    if len(X) % 2 == 0:
        mid = len(X) // 2
        pad = lambda X: F.pad(X, (0, 1), "constant", 0)  # noqa: E731
    else:
        mid = (len(X) + 1) // 2 - 1
        pad = lambda X: X  # noqa: E731

    Γ = F.conv1d(
        X.view(1, 1, -1),
        pad(X.view(1, 1, -1)),
        padding="same",
    ).squeeze()

    # print(Γ[mid - 1], Γ[mid], Γ[mid + 1])

    Γ = Γ[mid:] / Γ[mid]

    return Γ


compute_autocorrelation = torch.vmap(_compute_autocorrelation)


class IntegratedAutocorrelationErrors(NamedTuple):
    stat: Tensor
    bias: Tensor
    grad_stat: Tensor
    grad_bias: Tensor


@dataclass(frozen=True)
class ComputedAutocorrelations:
    autocorrelation: Tensor
    integrated: float
    truncation_window: int
    errors: IntegratedAutocorrelationErrors


def autocorrelations(X: Tensor, λ: float = 2.0) -> ComputedAutocorrelations:
    X = X.squeeze(-1)
    assert X.dim() == 2

    N = X.shape[1]  # sample size
    W = torch.arange(N // 2 + N % 2)  # compute window

    # Average over replica
    # NOTE use nanmean, which might hide issues
    Γ = compute_autocorrelation(X).nanmean(dim=0)

    # The summation window-dependent integrated autocorrelation
    τ_int = Γ.cumsum(dim=0) - 0.5

    # The associated exponential autocorrelation time
    τ_exp = ((2 * τ_int - 1) / (2 * τ_int + 1)).log().reciprocal().negative()
    τ_exp = τ_exp.nan_to_num().clamp(min=1e-6)

    # Statistical error (Eq. 42 in arxiv.org/pdf/hep-lat/0306017)
    ε_stat = torch.sqrt((4 / N) * (W + 1 / 2 - τ_int)) * τ_int

    # Truncation bias
    ε_bias = -τ_int * torch.exp(-W / (λ * τ_exp))

    # λ times W-derivative of the errors
    dεdW_stat = τ_exp / torch.sqrt(N * W)
    dεdW_bias = (-1 / λ) * torch.exp(-W / (λ * τ_exp))

    errors = IntegratedAutocorrelationErrors(
        stat=ε_stat,
        bias=ε_bias,
        grad_stat=dεdW_stat,
        grad_bias=dεdW_bias,
    )

    # Derivative of the sum of absolute errors
    dεdW = errors.grad_stat + errors.grad_bias

    # argmax returns first occurrence of the derivative being positive,
    # indicating that the total error will increase for larger window sizes
    W_opt = torch.argmax((dεdW[1:] > 0).int(), dim=0) + 1

    # Select the best estimate of the integrated autocorrelation time
    τ_int = τ_int[W_opt].item()

    return ComputedAutocorrelations(
        autocorrelation=Γ,
        integrated=τ_int,
        truncation_window=W_opt,
        errors=errors,
    )
