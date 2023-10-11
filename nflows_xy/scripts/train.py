from pathlib import Path
import logging

from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionYesNo,
    class_from_function,
    Namespace,
)
from jsonargparse.typing import Path_dc
import torch

from nflows_xy.core import Flow, FlowBasedSampler
from nflows_xy.xy import action
from nflows_xy.train import train, test
from nflows_xy.scripts.io import save_model
from nflows_xy.plot import plot_training_metrics_txt, plot_test_metrics_txt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(prog="train")
parser.add_argument("--flow", type=Flow)
parser.add_class_arguments(class_from_function(action), "target")
parser.add_function_arguments(train, "train", skip=["model"])
parser.add_function_arguments(test, "test", skip=["model"])
parser.add_argument("--cuda", action=ActionYesNo, help="train using CUDA")
parser.add_argument(
    "--double", action=ActionYesNo, help="use double precision"
)
parser.add_argument(
    "-o", "--output", type=Path_dc, help="location to save trained model"
)
parser.add_argument("-c", "--config", action=ActionConfigFile)


def main(config: Namespace) -> None:
    config_yaml = parser.dump(config, skip_none=False)

    config = parser.instantiate_classes(config)

    if config.output is None:
        logger.warning(
            "No output directory specified: trained model will not be saved!"
        )
        output_path = None

    else:
        output_path = Path(config.output)
        if output_path.exists():
            raise FileExistsError(f"{output_path} already exists!")

    model = FlowBasedSampler(config.flow, config.target)

    device = "cuda" if config.cuda else "cpu"
    dtype = torch.float64 if config.double else torch.float32
    model = model.to(device=device, dtype=dtype)

    training_metrics = train(model, **config.train)

    logger.info("Plotting training metrics...")
    figs = plot_training_metrics_txt(training_metrics)
    print("\n\n".join(list(figs.values())))
    
    test_metrics = test(model, **config.test)

    logger.info("Plotting test metrics...")
    figs = plot_test_metrics_txt(test_metrics)
    print("\n\n".join(list(figs.values())))

    print(test_metrics.describe())

    if output_path is not None:
        save_model(output_path, model, config_yaml)
