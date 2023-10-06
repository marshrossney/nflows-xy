from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import Path_dw

from quantum_rotor.action import PullbackAction
from quantum_rotor.hmc import hmc
from quantum_rotor.scripts.io import load_model

parser = ArgumentParser(prog="hmc")

parser.add_argument("model", type=Path_dw, help="path to a trained model")
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    model = load_model(config.model)
    print(model)

    action = PullbackAction(model.target, model.flow)

    _ = hmc(action, **config.hmc)
