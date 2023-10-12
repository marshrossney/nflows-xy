from math import pi as π

import plotille
import torch

from nflows_xy.autocorr import ComputedAutocorrelations
from nflows_xy.utils import mod_2pi

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

def plot_observable(
    observable: Tensor,
    autocorrelations: ComputedAutocorrelations,
    label: str,
) -> dict[str, str]:
    X = observable.squeeze(-1)  # scalars_only
    Γ = autocorrelations.autocorrelation
    τ = 0.5 * Γ.cumsum(0)
    stat = autocorrelations.errors.stat.abs()
    bias = autocorrelations.errors.bias.abs()
    W_opt = autocorrelations.truncation_window
    i_cut=2 * torch.argmax((Γ < 0).int())
    log_Γ = torch.tensor(Γ[:i_cut]).log()
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
    i_cut = i_cut.item()

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
    fig.plot(range(i_cut), stat[:i_cut], lc="blue", label="stat")
    fig.plot(range(i_cut), bias[:i_cut], lc="yellow", label="bias")
    fig.plot(range(i_cut), total[:i_cut], lc="green", label="total")
    fig.axvline(W_opt / i_cut, lc="red")
    dict_of_figs["errors"] = fig.show(legend=True)

    return dict_of_figs

