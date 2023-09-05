# emceeX

This project is a customization of the Markov chain Monte Carlo code [emcee](https://github.com/dfm/emcee/).  
Since I noticed I'm doing the same steps over and over again when running a MCMC model I've built a wrapper around `emcee` that can also be run from the command line.

It does not include all features of `emcee` and is not thoroughly tested. Use with caution and cross check your results.

## Installation

Clone the repository and install with

`pip install .`

## Usage

Usage is summarized in example notebooks.

1. [Basic usage](https://github.com/stammler/emceeX/blob/main/examples/1_linear_model/1_linear_model.ipynb)
2. [Command line](https://github.com/stammler/emceeX/blob/main/examples/2_command_line/2_command_line.ipynb)
3. [Monitoring](https://github.com/stammler/emceeX/blob/main/examples/3_monitoring/3_monitoring.ipynb)

For a complete manual on `emcee`, please have a look at the [documentation](https://emcee.readthedocs.io/) there.
