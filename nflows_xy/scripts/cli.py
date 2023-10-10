from jsonargparse import ArgumentParser

import nflows_xy.scripts.train as train
import nflows_xy.scripts.test as test
import nflows_xy.scripts.hmc as hmc
import nflows_xy.scripts.benchmark as benchmark

parser = ArgumentParser(prog="cli")

subcommands = parser.add_subcommands()
subcommands.add_subcommand("train", train.parser)
subcommands.add_subcommand("test", test.parser)
subcommands.add_subcommand("hmc", hmc.parser)
subcommands.add_subcommand("benchmark", benchmark.parser)


def cli():
    config = parser.parse_args()

    if config.subcommand == "train":
        train.main(config.train)
    elif config.subcommand == "test":
        test.main(config.test)
    elif config.subcommand == "hmc":
        hmc.main(config.hmc)
    elif config.subcommand == "benchmark":
        benchmark.main(config.benchmark)


if __name__ == "__main__":
    cli()