from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    Namespace,
)

from nflows_xy.hmc import hmc
from nflows_xy.autocorr import autocorrelations
from nflows_xy.xy import Action, top_charge
from nflows_xy.plot import plot_topological_charge


parser = ArgumentParser(prog="benchmark")

parser.add_class_arguments(Action, "action")
parser.add_function_arguments(hmc, "hmc", skip=["action"])

parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    config = parser.instantiate_classes(config)

    φ = hmc(config.action, **config.hmc)

    Q = top_charge(φ)
    Γ = autocorrelations(Q)

    print(Γ.integrated)

    figs = plot_topological_charge(Q, Γ)

    for k, v in figs.items():
        v.savefig(f"{k}.png")
