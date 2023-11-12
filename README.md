# Trivialising maps for the classical XY model

## Installation on Linux

1. Install the prerequisites
    - [pyenv](https://github.com/pyenv/pyenv) (e.g. using [this installer](https://github.com/pyenv/pyenv-installer))
    - [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)


2. Clone the repository

```sh
$ git clone https://github.com/marshrossney/nflows-xy
$ cd nflows-xy
```

3. Make sure pyenv knows to use Python 3.10.7 within this directory

```sh
$ pyenv install 3.10.7
$ pyenv local 3.10.7  # or pyenv global 3.10.7
```

4. Install the package and its dependencies

```sh
$ poetry install
```

## Usage

There is a CLI which is accessed via the command `nfxy`.
Generally speaking, commands are run in the following way:

```sh
$ poetry run nfxy SUBCOMMAND -c CONFIG_FILE_PATH --ADDITIONAL_ARGS
```

The important subcommands are
- `train`: Trains a new model from scratch
- `fhmc`: Use the trained model in the [flow-HMC algorithm](https://arxiv.org/abs/2302.08408) (often called 'field transformation' HMC).

Other subcommands are
- `test`: Computes some test metrics on a trained model. These metrics are computed whenever `train` or `fhmc` are run, but it can be useful to run them separately
- `hmc`: This just runs HMC without any flow/field transformation.

Example `.yaml` configuration files can be found in the `examples` folder.

To avoid cluttering up the terminal with tables and plots, you may want to redirect `stdout` to a file, e.g.

```sh
$ poetry run nfxy train -c /path/to/config.yaml > training.log
```

## Getting help

There are multiple ways to bring up help pages:

For the entire CLI...
```sh
$ poetry run nfxy --help
usage: nfxy [-h] {train,test,hmc,hmc-benchmark} ...

options:
  -h, --help            Show this help message and exit.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    train
    test
    hmc
    fhmc
```

For a specific subcommand...
```sh
$ poetry run nfxy hmc --help
usage: nfxy [options] hmc [-h] [--target CONFIG] --target.beta BETA --target.lattice_size LATTICE_SIZE --target.lattice_dim LATTICE_DIM [--hmc CONFIG] --hmc.n_replica N_REPLICA
                                         --hmc.n_traj N_TRAJ --hmc.step_size STEP_SIZE [--hmc.n_therm N_THERM] [--hmc.traj_length TRAJ_LENGTH] [-c CONFIG] [--print_config[=flags]] [-o OUTPUT]

Run a 'vanilla' Hybrid Monte Carlo simulation.

options:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are:
                        comments, skip_default, skip_null.
  -o OUTPUT, --output OUTPUT
                        location to save outputs (type: Path_dc, default: null)

Specify an XY action that can be used as a training or HMC target:
  --target CONFIG       Path to a configuration file.
  --target.beta BETA    Inverse temperature parameter, (β) (required, type: float)
  --target.lattice_size LATTICE_SIZE
                        Number of lattice sites in each dimension (required, type: int)
  --target.lattice_dim LATTICE_DIM
                        Number of lattice dimensions (1 or 2) (required, type: int)

Run the Hybrid Monte Carlo algorithm:
  --hmc CONFIG          Path to a configuration file.
  --hmc.n_replica N_REPLICA
                        Number of replica simulations running in parallel. (required, type: int)
  --hmc.n_traj N_TRAJ   Number of molecular dynamics trajectories. (required, type: int)
  --hmc.step_size STEP_SIZE
                        Molecular dynamics timestep. (required, type: float)
  --hmc.n_therm N_THERM
                        Number of trajectories to discard as 'thermalisation'. (type: int, default: 0)
  --hmc.traj_length TRAJ_LENGTH
                        Total time for each molecular dynamics trajectory. (type: float, default: 1.0)
```

Or for a specific class (in this case `AutoregressiveFlow`)
```sh
$ poetry run nfxy train --model.help AutoregressiveFlow
usage: nfxy --model.init_args.lattice_size LATTICE_SIZE --model.init_args.n_mixture N_MIXTURE --model.init_args.net_shape NET_SHAPE [--model.init_args.net_activation NET_ACTIVATION]
            [--model.init_args.weighted {true,false}] [--model.init_args.min_weight MIN_WEIGHT] [--model.init_args.ramp_pow RAMP_POW]

One-dimensional autoregressive flow model:
  --model.init_args.lattice_size LATTICE_SIZE
                        Number of lattice sites. (required, type: int)
  --model.init_args.n_mixture N_MIXTURE
                        Each layer is a convex combination of `n_mixture` transformations. (required, type: int)
  --model.init_args.net_shape NET_SHAPE, --model.init_args.net_shape+ NET_SHAPE
                        Hidden layer widths for the fully-connected neural networks. (required, type: list[int])
  --model.init_args.net_activation NET_ACTIVATION
                        Activation function to apply after each linear transformation. (type: str, default: Tanh)
  --model.init_args.weighted {true,false}
                        Whether or not the convex combinations are weighted mixtures. (type: bool, default: True)
  --model.init_args.min_weight MIN_WEIGHT
                        Minimum weight for weighted mixtures. (type: float, default: 0.01)
  --model.init_args.ramp_pow RAMP_POW
                        Integer power in the ramp function used to construct sigmoid transformations. (type: int, default: 2)
```

For more details see the [jsonargparse docs](https://jsonargparse.readthedocs.io/en).

## To do list

- [ ] Cauchy distributed momenta
- [ ] Reduced variance loss function (requires inverse of sigmoid transformations)
- [ ] Rational Quadratic Spline transformations with inverse
