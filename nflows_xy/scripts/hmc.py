from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from jsonargparse.typing import Path_dw

from nflows_xy.autocorr import autocorrelations
from nflows_xy.core import PullbackAction
from nflows_xy.hmc import hmc
from nflows_xy.scripts.io import load_model
from nflows_xy.xy import top_charge

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
