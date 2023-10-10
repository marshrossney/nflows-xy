from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import Path_dw

from quantum_rotor.autocorr import autocorrelations
from quantum_rotor.core import PullbackAction
from quantum_rotor.hmc import hmc
from quantum_rotor.scripts.io import load_model
from quantum_rotor.xy import top_charge

parser = ArgumentParser(prog="hmc")

parser.add_argument("model", type=Path_dw, help="path to a trained model")
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    model = load_model(config.model)
    # _ = model(1)
    print(model)

    action = PullbackAction(model.flow, model.target)

    sample = hmc(action, **config.hmc)

    Q = top_charge(sample)
    Γ = autocorrelations(Q)

    print(Γ.integrated)
