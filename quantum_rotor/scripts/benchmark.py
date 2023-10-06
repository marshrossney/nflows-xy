from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionYesNo,
    Namespace,
)

from quantum_rotor.action import ActionPBC, ActionOBC
from quantum_rotor.hmc import hmc
from quantum_rotor.topology import top_charge, autocorrelations
from quantum_rotor.plot import plot_topological_charge


parser = ArgumentParser(prog="benchmark")

parser.add_argument("--beta", type=float)
parser.add_argument("--obc", action=ActionYesNo)
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    action_cls = ActionOBC if config.obc else ActionPBC
    action = action_cls(config.beta)

    φ = hmc(action, **config.hmc)

    Q = top_charge(φ)
    Γ = autocorrelations(Q)

    print(Γ.integrated)

    figs = plot_topological_charge(Q, Γ)

    for k, v in figs.items():
        v.savefig(f"{k}.png")
