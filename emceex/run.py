import click
import dill
import emcee
from emceex import EnsembleSampler
import importlib
import numpy as np
import os
from pathlib import Path
import yaml

@click.command()
@click.argument("yaml_file", nargs=1, type=click.File("r"))
@click.option("-r", "--restart", "restart", is_flag=True, show_default=True, default=False, help="Delete dump file and restart model.")
@click.option('-v', '--verbose', default=0, count=True, show_default=True, help="Verbosity")
def run(yaml_file, restart, verbose):
    """
    Function reads the .yaml configuration file and starts `emceex.EnsembleSampler`.
    
    Parameters
    ----------
    yaml_file : filestream
        Stream of .yaml file
    restart : boolean
        If True sampler is forced to restart instead of resuming
    verbose : int
        Verbosity of run.
    """

    # Reading yaml file
    if verbose:
        print(f"Reading '{yaml_file.name}'.")
    dct = yaml.safe_load(yaml_file)
    
    # Reading dumpfile and deleting it if restarting
    resume = False
    dumpfile = dct.pop("dumpfile", None)
    if dumpfile is not None:
        dumpfile = Path(dumpfile)
        # Check if file exists
        if dumpfile.is_file():
            # Remove file if restart flag given
            if restart:
                if verbose:
                    print(f"Removing '{dumpfile}'.")
                dumpfile.unlink()
            # Load file, if restart flag not given
            else:
                if verbose:
                    print(f"Loading {dumpfile}.")
                mcmc = EnsembleSampler.load_dump(dumpfile)
                resume = True

    # If we are not resuming, read required information
    if not resume:
        
        # Reading paramters
        p = dct.pop("parameters", None)
        if p is None:
            raise RuntimeError("YAML file does not contain 'parameters'.")
        pars = {}
        for name, d in p.items():
            pmin = d.pop("min", None)
            pmax = d.pop("max", None)
            plog = d.pop("log", None)
            if pmin is None:
                raise RuntimeError(f"Parameter '{name}' does not contain 'min' value.")
            if pmax is None:
                raise RuntimeError(f"Parameter '{name}' does not contain 'max' value.")
            if plog is None:
                if verbose:
                    print(f"Parameter '{name}' does not contain 'log' flag. Assuming False.")
                plog = False
            pars[name] = {
                "min": float(pmin),
                "max": float(pmax),
                "log": plog,
            }
        N_dims = len(pars)
    
        # Reading data file
        datafile = dct.pop("datafile", None)
        if datafile is None:
            raise RuntimeError("No data file given.")
        datafile = Path(datafile)
        if verbose:
            print(f"Reading '{datafile}'.")
        data = np.loadtxt(datafile)
        x, y = data[:, 0], data[:, 1]
        if data.shape[1] == 3:
            yerr = data[:, 2]
        else:
            if verbose:
                print(f"Data file '{datafile}' not containing y error.")
            yerr = None
    
        # Number of walkers
        N_walkers = dct.pop("N_walkers", None)
        if N_walkers is None:
            N_walkers = 3*N_dims
            if verbose:
                print(f"'N_walkers' not given. Assuming 3*N_dims={N_walkers}.")
                
        # Loading model function
        model = dct.pop("model", None)
        if model is None:
            raise RuntimeError("No model given.")
        model_file = model.pop("file", None)
        if model_file is None:
            raise RuntimeError("No model file given.")
        model_function = model.pop("func", None)
        if model_function is None:
            raise RuntimeError("No model function given.")
        if verbose:
            print(f"Loading model function '{model_function}' from '{model_file}'.")
        spec = importlib.util.spec_from_file_location("model", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_fn = model_module.__dict__[model_function]
        model_kwargs = model.pop("kwargs", {})
        
        # Additional keywords arguments passed to sampler
        sampler_kwargs = dct.pop("sampler_kwargs", {})
        
    # Convergence options
    conv = dct.pop("convergence", {})
    beta = conv.pop("beta", 100.)
    delta = conv.pop("delta", 1.e-2)    
    
    # Number of steps
    N_steps = dct.pop("N_steps", None)
    if N_steps is None:
        N_steps = 10_000
        if verbose:
            print(f"'N_steps' not given. Assuming {N_steps}.")
            
    # Thin by
    thin_by = dct.pop("thin_by", 1)
    
    # Intervall
    interval = dct.pop("interval", 100)
            
    # Number of threads
    N_threads = dct.pop("N_threads", None)
    
    # Moves
    moves_dct = dct.pop("moves", None)
    if moves_dct is None:
        moves = None
    else:
        moves = []
        for name, val in moves_dct.items():
            move = (
                emcee.moves.__dict__[name](),
                val
            )
            moves.append(move)
    
    # Create Sampler object if not resuming
    if not resume:
        mcmc = EnsembleSampler(
            N_walkers,
            pars,
            model_fn,
            x, y, yerr,
            savepath=dumpfile,
            model_kwargs=model_kwargs,
            moves=moves,
            **sampler_kwargs,
        )
    # If resuming we may want to exchange moves
    else:
        if moves:
            mcmc._moves, mcmc._weights = zip(*moves)
            mcmc._weights /= np.sum(mcmc._weights)
            
    # Print information
    if verbose: 
        print()
        if resume:
            print("Resuming...")
        else:
            print("Starting...")
        print(f"# dimensions:    {mcmc.ndim:6d}")
        print(f"# walkers:       {mcmc.nwalkers:6d}")
        print(f"# threads:       {N_threads:6d}")
        print()
        if dumpfile:
            print(f"Saving to '{dumpfile}'")
            print(f"    with interval//thin_by = {interval//thin_by}.")
        print()
        
    
    # Run the model
    mcmc.run_mcmc(
        N_steps//thin_by,
        interval=interval//thin_by,
        nthreads=N_threads,
        beta=beta,
        delta=delta*thin_by,
        thin_by=thin_by,
        verbose=verbose,
    )        

if __name__ == "__main__":
    run()