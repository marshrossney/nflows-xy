import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import torch

from quantum_rotor.topology import ComputedAutocorrelations

Tensor = torch.Tensor
Figure = plt.Figure

plt.style.use("seaborn-v0_8-darkgrid")


def plot_topological_charge(
    Q: Tensor,
    autocorrelations: ComputedAutocorrelations,
) -> dict[str, Figure]:
    dict_of_figs = {}

    # Plot topological charge time series
    n_replica, n_traj, _ = Q.shape
    n_to_plot = min(n_replica, 4)
    Q_to_plot = Q[:n_to_plot]
    fig, axes = plt.subplots(n_to_plot, 1, sharex=True, sharey=True)
    axes[-1].set_xlabel(r"$t$")
    axes[0].set_title("Topological charge time series")
    for q, ax in zip(Q_to_plot, axes):
        ax.plot(q.squeeze(-1))
    dict_of_figs |= {"top_charge": fig}

    # Plot autocorrelation
    Γ = autocorrelations.autocorrelation
    W_opt = autocorrelations.truncation_window
    i_cut = 2 * torch.argmax((Γ < 0).int())
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\delta t$")
    ax.set_ylabel(r"$\Gamma(\delta t)$")
    ax.set_title("Autocorrelation of $Q$")
    ax.plot(Γ[:i_cut], label=r"$\Gamma(\delta t)$")
    ax.axvline(W_opt, linestyle="--", color="red", label=r"$W_{opt}$")
    ax.set_yscale("log")
    ax.legend()
    dict_of_figs |= {"autocorrelation": fig}

    # Plot integrated autocorrelation
    τ = 0.5 * Γ.cumsum(0)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"$\tau(W)$")
    ax.set_title("Integrated autocorrelation of $Q$")
    ax.plot(
        τ[:i_cut],
        label=r"$\tau(W) = \frac{1}{2} \sum_{\delta t = 0}^{W} \Gamma(\delta t)$",
    )
    ax.axvline(W_opt, linestyle="--", color="red", label=r"$W_{opt}$")
    ax.legend()
    dict_of_figs |= {"integrated": fig}

    # Plot errors
    errors = autocorrelations.errors
    stat = errors.stat.abs()
    bias = errors.bias.abs()
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"Errors on $\tau(W)$")
    ax.set_title("Errors on integrated autocorrelation of $Q$")
    ax.plot(stat[:i_cut], label=r"$|\epsilon_{stat}|$")
    ax.plot(bias[:i_cut], label=r"$|\epsilon_{bias}|$")
    ax.plot(
        (stat + bias)[:i_cut], label=r"$|\epsilon_{stat}| + |\epsilon_{bias}|$"
    )
    ax.axvline(W_opt, linestyle="--", color="red", label=r"$W_{opt}$")
    ax.legend()
    dict_of_figs |= {"errors": fig}

    return dict_of_figs
