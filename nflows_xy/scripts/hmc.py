from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import logging

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    class_from_function,
    Namespace,
)
from jsonargparse.typing import Path_dc
import pandas as pd

from nflows_xy.autocorr import autocorrelations
from nflows_xy.hmc import hmc
from nflows_xy.plot import plot_observable, plot_spin_correlation
from nflows_xy.xy import action, top_charge
from nflows_xy.scripts.io import SamplingDirectory
from nflows_xy.utils import make_banner

from nflows_xy.xy import spin_correlation, fit_spin_correlation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(
    prog="hmc",
    description="Run a 'vanilla' Hybrid Monte Carlo simulation.",
)

parser.add_class_arguments(class_from_function(action), "target")
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)
parser.add_argument(
    "-o", "--output", type=Path_dc, help="location to save outputs"
)


def main(config: Namespace) -> None:
    config_copy = deepcopy(config)

    if config.output is None:
        logger.warning(
            "No output directory specified: Sample outputs will not be saved!"
        )
        output_path = None

    else:
        output_path = Path(config.output)
        if output_path.exists():
            raise FileExistsError(f"{output_path} already exists!")

    config = parser.instantiate_classes(config)

    φ, metrics = hmc(config.target, **config.hmc)

    S = config.target(φ)
    Γ_S = autocorrelations(S)
    logger.info("Plotting the action...")
    figs = plot_observable(S, Γ_S, "S")

    print(make_banner("Action plots"))
    print("\n".join(list(figs.values())))

    Q = top_charge(φ)
    Γ_Q = autocorrelations(Q)
    print("Q: ", Q.shape)
    logger.info("Plotting the topological charge...")
    figs = plot_observable(Q, Γ_Q, "Q")

    print(make_banner("Topological Charge plots"))
    print("\n".join(list(figs.values())))

    G = spin_correlation(φ)
    (ξ, c), cov = fit_spin_correlation(G)
    logger.info("Plotting the spin correlation function...")
    fig = plot_spin_correlation(G.mean(0).log(), ξ, c)

    print(make_banner("Spin correlation"))
    print(fig)

    metrics = pd.Series(
        asdict(metrics)
        | {
            "corr_len": ξ.item(),
            "tau_int_S": Γ_S.integrated,
            "tau_int_Q": Γ_Q.integrated,
        }
    )
    print(make_banner("Metrics"))
    print(metrics)

    if output_path is not None:
        _ = SamplingDirectory.new(
            output_path,
            sample=φ,
            config=config_copy,
            metrics=metrics,
        )
