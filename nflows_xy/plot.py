from math import pi as π

import plotille
import torch

from nflows_xy.autocorr import ComputedAutocorrelations
from nflows_xy.utils import mod_2pi
from nflows_xy.xy import log_cosh

Tensor = torch.Tensor

_plotille_colours = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
]


def plot_training_metrics(metrics):
    output = {}
    t_max = int(metrics.step.max())
    for metric in metrics.drop(columns="step"):
        fig = plotille.Figure()
        fig.x_label = "step"
        fig.y_label = metric
        fig.set_x_limits(min_=0, max_=t_max)
        fig.scatter(
            metrics["step"],
            metrics[metric],
        )
        output[metric] = fig.show()
    return output


def plot_test_metrics(metrics):
    output = {}
    for metric in metrics:
        fig = plotille.Figure()
        fig.x_label = metric
        fig.y_label = "count"
        fig.histogram(
            metrics[metric],
            bins=15,
        )
        output[metric] = fig.show()
    return output


def plot_spins(φ: Tensor):
    L = φ.shape[1]
    φ, _ = φ.tensor_split([L // 2 + 1], dim=-2)
    φ0, φ = φ.tensor_split([1], dim=-2)
    φ = mod_2pi(φ - φ0 + π) - π

    fig = plotille.Figure()
    fig.x_label = f"φx - φ0"
    fig.y_label = "count"
    fig.set_x_limits(min_=-π, max_=π)

    for x, φ_x in enumerate(φ.split(1, dim=-2)):
        hist, bin_edges = torch.histogram(
            φ_x.flatten(), bins=25, range=(-π, π)
        )
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        fig.plot(midpoints.tolist(), hist.tolist(), label=f"φ{x+1}")

    return fig.show(legend=True)


def plot_spin_correlation(
    log_G: Tensor,
    ξ: float,
    c: float,
):
    L = len(log_G)
    mask = log_G.isfinite()
    log_G = log_G[mask].tolist()
    x = torch.arange(L)[mask].tolist()

    xx = torch.linspace(0, L, 100)
    log_G_fit = log_cosh((xx - L / 2) / ξ) + c

    fig = plotille.Figure()
    fig.x_label = "δx"
    fig.y_label = "log G(δx)"
    fig.set_x_limits(min_=0, max_=L - 1)
    fig.plot(xx.tolist(), log_G_fit.tolist(), lc="red", label="fit")
    fig.scatter(x, log_G, lc="blue", marker="+", label="data")

    return fig.show(legend=True)


def plot_observable(
    observable: Tensor,
    autocorrelations: ComputedAutocorrelations,
    label: str,
) -> dict[str, str]:
    X = observable.squeeze(-1)  # scalars_only
    n_rep, n_samp = X.shape
    Γ = autocorrelations.autocorrelation
    τ = Γ.cumsum(0) - 0.5
    stat = autocorrelations.errors.stat.abs()
    bias = autocorrelations.errors.bias.abs()
    W_opt = autocorrelations.truncation_window

    # Select a cutoff δx for the plots
    # i_cut = min(torch.argmax((Γ < 0).int()).item(), n_samp // 2 - 1)
    i_cut = min(4 * W_opt.item(), n_samp // 2 - 1)
    log_Γ = Γ[:i_cut].log()
    mask = log_Γ.isfinite()
    log_Γ = log_Γ[mask].tolist()
    t = torch.arange(i_cut)[mask].tolist()

    X = X[0].tolist()
    Γ = Γ.tolist()
    τ = τ.tolist()
    total = (stat + bias).tolist()
    stat = stat.tolist()
    bias = bias.tolist()
    W_opt = W_opt.item()

    dict_of_figs = {}

    # Plot time series
    t_max = len(X)
    fig = plotille.Figure()
    fig.x_label = "t"
    fig.y_label = label
    fig.set_x_limits(min_=0, max_=t_max)
    fig.plot(range(t_max), X)
    dict_of_figs["time_series"] = fig.show()

    # Plot autocorrelation
    fig = plotille.Figure()
    fig.x_label = "δt"
    fig.y_label = f"[{label}] log Γ(δt)"
    fig.set_x_limits(min_=0, max_=i_cut)
    fig.plot(t, log_Γ)
    fig.scatter(t, log_Γ, lc="blue", marker="+")
    fig.axvline(W_opt / i_cut, lc="red")
    dict_of_figs["autocorrelation"] = fig.show()

    # Plot integrated autocorrelation
    fig = plotille.Figure()
    fig.x_label = "W"
    fig.y_label = f"[{label}] τ(W)"
    fig.set_x_limits(min_=0, max_=i_cut)
    fig.plot(range(i_cut), τ[:i_cut])
    fig.axvline(W_opt / i_cut, lc="red")
    dict_of_figs["integrated"] = fig.show()

    # Plot errors
    fig = plotille.Figure()
    fig.x_label = "W"
    fig.y_label = f"[{label}] ε(W)"
    fig.set_x_limits(min_=0, max_=i_cut)
    fig.plot(range(i_cut), stat[:i_cut], lc="blue", label="stat")
    fig.plot(range(i_cut), bias[:i_cut], lc="yellow", label="bias")
    fig.plot(range(i_cut), total[:i_cut], lc="green", label="total")
    fig.axvline(W_opt / i_cut, lc="red")
    dict_of_figs["errors"] = fig.show(legend=True)

    return dict_of_figs
