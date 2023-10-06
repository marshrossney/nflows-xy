from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionYesNo,
    Namespace,
)

from quantum_rotor.action import ActionPBC, ActionOBC
from quantum_rotor.hmc import hmc


parser = ArgumentParser(prog="benchmark")

parser.add_argument("--beta", type=float)
parser.add_argument("--obc", action=ActionYesNo)
parser.add_function_arguments(hmc, "hmc", skip=["action"])
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    action_cls = ActionOBC if config.obc else ActionPBC
    action = action_cls(config.beta)

    _ = hmc(action, **config.hmc)
