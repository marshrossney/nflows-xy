# Trivialising maps for the classical XY model

## Installation on Linux

### Prerequisites

1. [pyenv](https://github.com/pyenv/pyenv) (e.g. using [this installer](https://github.com/pyenv/pyenv-installer))
2. [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

### Installation

1. Clone the repository

```sh
$ git clone https://github.com/marshrossney/nflows-xy
$ cd nflows-xy
```

2. Make sure pyenv knows to use Python 3.10.7 within this directory

```sh
$ pyenv install 3.10.7
$ pyenv local 3.10.7  # or pyenv global 3.10.7
```

3. Install the package and its dependencies

```sh
$ poetry install
```

## Command-line interface

There is a CLI which is accessed via the command `nflows-xy`.
Generally speaking, commands are run in the following way:

```sh
$ poetry run nflows-xy SUBCOMMAND -c CONFIG_FILE_PATH --ADDITIONAL_ARGS
```

Example `.yaml` configuration files can be found in the `examples` folder.

### Help pages

There are multiple ways to bring up the help pages:

For the entire CLI...
```sh
$ poetry run nflows-xy --help
usage: nflows-xy [-h] {train,test,hmc,benchmark} ...

options:
  -h, --help            Show this help message and exit.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    train
    test
    hmc
    benchmark
```

For a specific subcommand...
```sh
$ poetry run nflows-xy train --help
usage: nflows-xy [options] train [-h] [--flow.help CLASS_PATH_OR_NAME] [--flow FLOW] [--target CONFIG]
                           --target.beta BETA --target.lattice_size LATTICE_SIZE
                           --target.lattice_dim LATTICE_DIM [--train CONFIG] --train.n_steps N_STEPS
                           --train.batch_size BATCH_SIZE [--train.init_lr INIT_LR]
                           [--train.metrics_interval METRICS_INTERVAL] [--cuda] [--double] [-o OUTPUT]
                           [-c CONFIG] [--print_config[=flags]]

options:
...
```

Or for a specific class (in this case `AutoregressiveFlow`)
```sh
$ poetry run  nflows-xy train --flow.help AutoregressiveFlow
usage: nflows-xy --flow.init_args.lattice_size LATTICE_SIZE --flow.init_args.n_mixture N_MIXTURE
                 --flow.init_args.net_shape NET_SHAPE [--flow.init_args.net_activation NET_ACTIVATION]
                 [--flow.init_args.weighted {true,false}] [--flow.init_args.min_weight MIN_WEIGHT]
                 [--flow.init_args.ramp_pow RAMP_POW]

<class 'nflows_xy.flows.AutoregressiveFlow'>:
...
```

For more details see the [jsonargparse docs](https://jsonargparse.readthedocs.io/en).
