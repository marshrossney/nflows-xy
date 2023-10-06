from jsonargparse import ArgumentParser

import quantum_rotor.scripts.train as train
import quantum_rotor.scripts.test as test
import quantum_rotor.scripts.hmc as hmc
import quantum_rotor.scripts.benchmark as benchmark

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
