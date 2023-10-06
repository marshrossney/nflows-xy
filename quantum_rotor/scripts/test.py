from jsonargparse import ArgumentParser, Namespace
from jsonargparse.typing import Path_dw

from quantum_rotor.train import test
from quantum_rotor.scripts.io import load_model

parser = ArgumentParser(prog="test")

parser.add_argument("model", type=Path_dw, help="path to a trained model")
parser.add_function_arguments(test, "test", skip=["model"])


def main(config: Namespace) -> None:
    model = load_model(config.model)

    metrics = test(model, **config.test)
    print(metrics)
