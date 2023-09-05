import cornerhex
import dill
import emcee
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import warnings


class EnsembleSampler(emcee.EnsembleSampler):
    """
    Wrapper for emcee.EnsembleSampler with some additional functionality for convenience.
    """
    
    def __init__(self, nwalkers, parameters, model_fnc, x, y, yerr=None, model_kwargs={}, savepath=None, **emcee_kwargs):
        """
        Wrapper for emcee.EnsembleSampler
        
        Parameters
        ----------
        nwalkers : int
            Number of walkers
        parameters : dict
            Dictionary of model parameters
        model_fnc : callable
            Model function
        x : array-like
            x values of data
        y : array-like
            y values of data
        yerr : array-like, optional
            y error of data. If not given, a constant error of unity
            is used in the computation.
        model_kwargs : dict, optional, default: {}
            Keyword arguments passed to model function
        savepath : str or Path, optional, default: None
            If given, dump files will be written to given path
        emcee_kwargs : Additional keyword arguments passed to emcee.EnsembleSampler.
            Not all possible keywords will function as intended.
        
        Information
        -----------
        
        The ``parameters`` dictionary has to have the following structure
        
        parameters {
            "parameter_name_1": {
                "min": float,   # Minimum value
                "max": float,   # Maximum value
                "log": boolean, # If True, parameter is sampled logarithmically
            },
            "parameter_name_2": {
                "min": float,
                "max": float,
                "log": boolean,
            },
            .
            .
            .
        }
        
        The model function needs to have the paramter vector as first and
        the x values as second argument.
        
        def model(theta, x):
            m, C = theta
            return m*x +C
        """
        
        # Random generator for picking initial values
        self.rng = np.random.default_rng()
        
        # Parsing the parameters dictionary and saving attributes
        self.bounds, self.log, self.labels = self._parse_parameters(parameters)
        
        # The model function
        self._model_fnc = model_fnc
        
        # The path to store dump files
        self.savepath = savepath
        if self.savepath is not None:
            self.savepath = Path(self.savepath)
            self.savepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Data
        # Use default y error of 1 if not given.
        self.x = x
        self.y = y
        self.yerr = np.ones_like(y) if yerr is None else yerr
        
        # Keyword arguments passed to model function
        self.model_kwargs = model_kwargs
        
        # These are stored later and are placeholder.
        # Used for monitoring.
        self.beta = None
        self.delta = None
        
        # Initializing the EnsembleSampler
        super(EnsembleSampler, self).__init__(
            nwalkers,                         # Number of walkers
            len(parameters),                  # Number of dimensions
            EnsembleSampler._log_probability, # Log probability function
            args=[                            # Arguments passed to log probability function
                self._model_fnc,              # of emcee.
                self.bounds,
                self.log,
                self.x, self.y, self.yerr,
                self.model_kwargs,
            ],
            **emcee_kwargs,                   # Additional keyword arguments
        )
    
    
    @staticmethod
    def _parse_parameters(pars):
        """
        Function takes a parameters dictionary and extracts necessary information.
        
        Parameters
        ----------
        pars : dict
            Dictionary with parameter information
            
        Returns
        -------
        bounds : array-like, (ndims, 2)
            Array with minimum/maximum bounds of model parameters
        log : array-like, (ndims,)
            Boolean array for logarithmic parameter sampling
        labels : array-like, (ndims,)
            List of strings with parameter names
        """
        bounds = np.empty((len(pars), 2))
        for i, (name, p) in enumerate(pars.items()):
            bounds[i, 0] = np.log10(p["min"]) if p["log"] else p["min"]
            bounds[i, 1] = np.log10(p["max"]) if p["log"] else p["max"]
        log = np.array([pars[p]["log"] for p in pars])
        labels = np.array([p for p in pars])
        return bounds, log, labels
    
    
    @staticmethod
    def _pick_ini(bounds, nwalkers, rng=None):
        """
        Function picks random initial position of walkers.
        
        Parameters
        ----------
        bounds : array-like, (ndims, 2)
            Array with parameter bounds
        nwalkers : int
            Number of walkers
        rng : numpy.random._generator.Generator, optional
            Generator to pick random numbers
            
        Returns
        -------
        p0 : array-like, (nwalkers, ndims)
            Initial walker positions
        """
        rng = rng or np.random.default_rng()
        ndims = bounds.shape[0]
        return (bounds[:, 1]-bounds[:, 0])[None, :]*rng.random((nwalkers, ndims)) + bounds[:, 0][None, :]
    
    
    @staticmethod
    def _log_likelihood(y_model, x_data, y_data, yerr_data):
        """
        Function computes log likelihood.
        
        Parameters
        ----------
        y_model : array-like
            y values of model
        x_data : array-like
            x values of data
        y_data : array-like
            y values of data
        yerr_data : array-like
            y error of data
            
        Returns
        -------
        log_likelihood : float
            Log likelihood of given values
        """
        return -0.5 * np.sum( ((y_model-y_data)/yerr_data)**2 + np.log(2*np.pi*yerr_data**2) )
    
    
    @staticmethod
    def _log_probability(theta, model, bounds, log, x_data, y_data, yerr_data, model_kwargs={}):
        """
        Function computes the log probability and is used in the
        emcee.EnsembleSampler object as model function. It needs
        access to the currect emceex object to access the actual
        model function and infrastructure.
        
        Parameters
        ----------
        theta : array-like, (ndims,)
            State vector
        model : callable
            Model function
        bounds : array-like, (ndims, 2)
            Array with parameter bounds
        log : array-like, (ndims,)
            Boolean array of logarithmic sampling
        x_data : array-like
            x values of data
        y_data : array-like
            y values of data
        yerr_data : array-like
            y error of data
        model_kwargs : dict, optional
            Keyword arguments passed to model function
            
        Returns
        -------
        log_probability : float
            Log probability of state vector
        """
        # Computing the log prior from static method of emceex
        # with state vector and bounds array from emceex instance
        log_prior = EnsembleSampler._log_prior(theta, bounds)
        # If the prior is not finite we can immediately
        # return -np.inf
        if not np.isfinite(log_prior):
            return -np.inf
        # The model function takes the state vector in actual space
        # while we store the parameters in logarithmic space if
        # requested. We therefore need to convert the state vector.
        p = np.where(log, 10.**theta, theta)
        # Computing the y values of the model given the parameter
        # state vector.
        y_model = model(p, x_data, **model_kwargs)
        # Computing the log likelihood using the static method
        # of the emceex instance
        log_likelihood = EnsembleSampler._log_likelihood(
            y_model,
            x_data, y_data, yerr_data
        )
        # Returning the log probability
        return log_prior + log_likelihood
    
    
    @staticmethod
    def _log_prior(theta, bounds):
        """
        Function computes the uninformed log prior and returns
        -np.inf if the state vector is out of bounds.
        
        Parameters
        ----------
        theta : array-like, (ndims,)
            State vector
        bounds : array-like, (ndims, 2)
            Lower and upper bounds of parameters
            
        Returns
        -------
        log_prior : float
            Log prior of state vector
        """
        mask_lower = (theta < bounds[:, 0])
        mask_upper = (theta > bounds[:, 1])
        if np.any(mask_lower) or np.any(mask_upper):
            return -np.inf
        return 0.
    
    
    @staticmethod
    def load_dump(path):
        """
        Loading sampler from dump file using dill.
        
        Parameters
        ----------
        path : str or Path
            Path to dump file
            
        Returns
        -------
        sampler : emceex.EnsembleSampler
            Loaded sampler
        """
        with open(path, "rb") as dumpfile:
            sampler = dill.load(dumpfile)
        # Put backend state in accordance with run of dumpfile
        # is from interrupted run
        sampler.backend.chain = sampler.backend.chain[:sampler.iteration, ...]
        sampler.backend.log_prob = sampler.backend.log_prob[:sampler.iteration, ...]
        return sampler
        
        
    @staticmethod
    def write_dump(path, sampler):
        """
        Writing sampler to dump file using dill.
        Make sure to unset the pool before calling this function.
        
        Parameters
        ----------
        path : str or Path
            Path to dump file
        sampler : emceex.EnsembleSampler
            Sampler to be stored
        """
        with open(path, "wb") as dumpfile:
            return dill.dump(sampler, dumpfile)
        
        
    def model(self, theta=None, x=None, model_kwargs=None):
        """
        Function evaluates the model function.
        
        Parameters
        ----------
        theta : array-like, (ndims,), optional, default: None
            Parameters state vector. Defaults to median values
            if not given
        x : array-like, optional, default: None
            x values of model function. Defaults to data values
            if not given.
        model_kwargs : dict, optional, default: None
            Keyword arguments passed to model function.
            Defaults used if not given.
        """
        # Getting median parameter vector if not given.
        if theta is None:
            theta = self.get_theta()
        # Getting x values if not given.
        if x is None:
            x = self.x
        # Getting model keyword arguments if not given.
        # If None we have to pass empty dictionary.
        if model_kwargs is None:
            model_kwargs = self.model_kwargs
        else:
            model_kwargs = {}
        return self._model_fnc(theta, x, **model_kwargs)
        
    
    def run_mcmc(self, nsteps, nthreads=0, interval=100, beta=100, delta=0.01, verbose=2, **emcee_kwargs):
        """
        Function is a wrapper for emcee to run the sampling.
        
        Parameters
        ----------
        nsteps : int
            Number of MCMC steps
        nthreads : int of None, optional, default: 0
            Number of threads used in mutiprocessing
        interval : int, optional, default: 100
            Checking for convergence after every interval step and
            writing of dump files
        beta : float, optional, default: 100.
            Convergence if chain is beta times longer than
            autocorrelation time.
        delta : float, optional, default: 0.01
            Maximum accepted change of the autorcorrelation
            time to accept convergence.
        verbose : int, optional, default: 2
            Verbosity option. Will show progress bar if larger
            than 1.
        emcee_kwargs: Additional keyword arguments passed to
            emcee.sample. Not all arguments will function.
        """
        # Storing convergence parameters for monitoring.
        self.beta = beta
        self.delta = delta
        self.interval = interval
        # If the sampler has not run before, the initial state
        # of the walkers are picked randomly. Otherwise the
        # last values of the chains are used.
        if self.iteration == 0:
            p0 = EnsembleSampler._pick_ini(self.bounds, self.nwalkers, self.rng)
        else:
            p0 = self.chain[:, -1, :]
        # If nthreads is not None a multiprocess.Pool is
        # created and stored in the object. If nthreads is
        # None the pool is unset to turn off multiprocessing.
        # The actual computation happens within EnsembleSampler._run().
        if nthreads:
            with mp.Pool(nthreads) as pool:
                self.pool = pool
                self._run(p0, nsteps, verbose=verbose, **emcee_kwargs)
        else:
            self.pool = None
            self._run(p0, nsteps, verbose=verbose, **emcee_kwargs)
            
            
            
    def _run(self, p0, nsteps, verbose=2, **emcee_kwargs):
        """
        Function runs the MCMC sampling.
        
        Parameters
        ----------
        p0 : array-like, (nwalkers, ndims)
            Initial state of parameters vector
        nsteps : int
            Number of sampling steps
        verbose : int, optional, default : 2
            Verbosity option. Will show progress bar if larger
            than 1.
        emcee_kwargs: Additional keyword arguments passed to
            emcee.sample. Not all arguments will function.
        """
        # Using emcee.sample to run the sampling
        sampler = super(EnsembleSampler, self).sample(
                    p0,
                    iterations=nsteps,
                    **emcee_kwargs
        )
        # Setting up the progress bar using tqdm.auto if requested.
        if verbose>1:
            sampler = tqdm(
                sampler,
                initial=self.iteration,                           # Initial value of the iteration
                                                                  # Can differ from 0 if run has been resumed
                total=self.iteration+nsteps,                      # End value of iteration
                desc="beta: N/A | delta : N/A",                   # Writing information in progress bar
                leave=True,                                       # Leave the progress bar on screen when finished
            )
        # Storing the old value the autocorrelation time to check for convergence.
        # Try to get initial tau if run is continued. Use infinity else
        tau_old = np.inf
        if self.iteration>0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tau = np.mean(self.get_autocorr_time(tol=0))
            if np.any(~np.isfinite(tau)):
                tau_old = np.inf
            else:
                tau_old = tau
        # Running the iteration
        for sample in sampler:
            # If we do not need to check for convergence or write dump file
            # simply continue with the next iteration,
            if self.iteration%self.interval:
                continue
            # Get the autocorrelation time. Ignore errors and warning
            # if chain too short.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                tau = np.max(self.get_autocorr_time(tol=0))
            # Normalized tau to beta
            tau_n = self.beta*tau / self.iteration
            # Relative change in autocorrelation time
            delta_tau = np.abs(tau_old-tau)/tau/self.delta
            # Update the description of the progress bar.
            if verbose>1:
                if np.isfinite(tau_n) and np.isfinite(delta_tau):
                    desc = "beta: {:.3f} | delta: {:.3f}".format(tau_n, delta_tau)
                else:
                    desc = "beta: N/A | delta: N/A"
                sampler.set_description(desc)
            # Write dump file if path is given.
            # Do not dump the pool! This will destroy multiprocessing!
            if self.savepath is not None:
                pool = self.pool
                self.pool = None
                EnsembleSampler.write_dump(self.savepath, self)
                self.pool = pool
            # Check for convergence. If converged close the progress bar and
            # leave the iteration.
            converged = (tau_n<=1.) & (delta_tau<=1.)
            if converged:
                if verbose>1:
                    sampler.close()
                break
            # If not converged save the value of tau
            tau_old = tau
        # If run within a notebook, change color of progress bar to green.
        # Would be red otherwise after break.
        if hasattr(sampler, "container"):
            sampler.container.children[1].bar_style = "success"

                
    def get_flat_samples(self, discard=None, thin=None):
        """
        Wrapper function to get flattened samples with
        reasonable assumption on discard and thin.
        
        Parameters
        ----------
        discard : int, optional, default: None
            Discard the first discard steps. Defaults to
            three times the autocorrelation time.
        thin : int, optional, default: None
            Thin samples by thin. Defaults to half of the
            autocorrelation time.
            
        Returns
        -------
        samples : array-like, (N, ndims)
            Flattened samples
            
        Information
        -----------
        If discard is not given, three times of the
        autocorrelation time will be discarded.
        If thin is not given, the samples will be
        thinned by half of the autocorrelation time.
        The samples will be returned in actual parameters
        space. Not in logarithmic space. If the
        autocorrelation time is not finite discard will
        be set to zero and thin to one.
        """
        # Getting the autocorrelation time.
        # This can raise warnings if the chain is too short.
        # We ignore them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tau = np.mean(self.get_autocorr_time(tol=0))
        # If the autocorrelation time is not finite and discard
        # or thin are not set, they are manually set here.
        if np.any(~np.isfinite(tau)):
            if discard is None:
                discard = 0
            if thin is None:
                thin = 1
        else:
            if discard is None:
                discard = int(tau*3)
            if thin is None:
                thin = int(tau//2)
        flat_samples = self.get_chain(discard=discard, thin=thin, flat=True)
        flat_samples = np.where(self.log, 10.**flat_samples, flat_samples)
        return flat_samples
        
        
    def plot_walkers(self, width=6., alpha=0.1, theta=None, theta_labels=None):
        """
        Function plots the walkers for inspection.
        
        Parameters
        ----------
        width : float, optional, default: 6.
            Width of plot
        alpha : float, optional, default: 0.1
            Alpha transparency of individual walkers
        theta : array-like, (ndims,) or list, optional, default: None
            Parameter values to be highlighted
        theta_labels : str of list, optional, default: None
            Labels of theta values to be plotted
            in the legend
        
        Returns
        -------
        fig, ax : Figure and Axes objects
        """
        # Dimensions of a single plot
        height = width/6.
        # Get all walker chains
        samples = self.get_chain()
        # Convert chains to actual parameter space
        samples = np.where(self.log[None, None, :], 10.**samples, samples)
        fig, ax = plt.subplots(sharex=True, nrows=self.ndim, figsize=(width, self.ndim*height))
        x = np.arange(samples.shape[0])
        for i in range(self.ndim):
            ax[i].plot(x, samples[:, :, i], lw=1, alpha=alpha, c="black")
            # Highlight one or more theta values with labels
            if theta is not None:
                if len(np.array(theta).shape)>1:
                    for j, t in enumerate(theta):
                        label = None
                        if theta_labels:
                            label = theta_labels[j]
                        ax[i].plot(x, np.ones_like(x)*t[i], lw=1, label=label)
                else:
                    ax[i].plot(x, np.ones_like(x)*theta[i], lw=1, label=theta_labels)
            ax[i].set_ylabel(self.labels[i])
            # Change to log scale if parameter has been sampled logarithmically
            if self.log[i]:
                ax[i].set_yscale("log")
        ax[-1].set_xlabel("Steps")
        ax[-1].set_xlim(x.min(), x.max())
        if theta_labels:
            ax[0].legend(ncols=len(theta_labels), fontsize="x-small")
        fig.tight_layout()
        return fig, ax
    
    
    def plot_data(self, width=6., height=3.75, theta=None, theta_labels=None, nsamples=0, alpha=0.1, discard=None, thin=None):
        """
        Function plots the data points.
        
        Parameters
        ----------
        width : float, optional, default: 6.
            Width of plot
        width : float, optional, default: 3.75
            Height of plot
        theta : array-like, (ndims,) or list, optional, default: None
            Parameter values to be highlighted
        theta_labels : str of list, optional, default: None
            Labels of theta values to be plotted
            in the legend
        nsamples : int, optional, default: 0
            Number of random samples to be plotted
        alpha : float, optional, default: 0.1
            Alpha transparency value of samples
        discard : int, optional, default: None
            Discard the first discard steps when picking random samples.
            Defaults to three times the autocorrelation time.
        thin : int, optional, default: None
            Thin samples by thin when picking random samples. Defaults
            to half of the autocorrelation time.
            
        Returns
        -------
        fig, ax : Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(width, height))
        # Plot the actual data
        ax.errorbar(self.x, self.y, yerr=self.yerr, marker=".", linestyle="None", color="black", capsize=3, lw=1)
        # Plot nsamples random sample from walkers
        if nsamples:
            flat_samples = self.get_flat_samples(discard=discard, thin=thin)
            inds = np.random.randint(len(flat_samples), size=nsamples)
            for i in inds:
                ax.plot(self.x, self.model(theta=flat_samples[i]), alpha=alpha, c="black", lw=1)
        # Highlight one or more theta values with labels
        if theta is not None:
            if len(np.array(theta).shape)>1:
                for i, t in enumerate(theta):
                    label = None
                    if theta_labels:
                        label = theta_labels[i]
                    ax.plot(self.x, self.model(theta=t), label=label, lw=1)
            else:
                ax.plot(self.x, self.model(theta=theta), label=theta_labels, lw=1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(self.x.min(), self.x.max())
        if theta_labels:
            ax.legend()
        fig.tight_layout()
        return fig, ax
    
    
    def plot_pairs(self, discard=None, thin=None, **cornerhex_kwargs):
        """
        Function plots corner plot using cornerhex.
        
        Parameters
        ----------
        discard : int, optional, default: None
            Discard the first discard steps when picking samples.
            Defaults to three times the autocorrelation time.
        thin : int, optional, default: None
            Thin samples by thin when picking samples. Defaults
            to half of the autocorrelation time.
        **cornerhex_kwargs : Keyword arguments passed to ``cornerhex.cornerplot()``
        
        Returns
        -------
        fig, ax : Figure and Axed objects
        """
        # Get the flat samples with reasonable discard and thin.
        flat_samples = self.get_flat_samples(discard=discard, thin=thin)
        # Convert to logspace if sampled logarithmically.
        # This has to be done, since cornerhex does not have
        # logarithmic axes.
        # Disable warning for this step since ``np.log10()`` might
        # get negative values for linearly sampled parameters.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            flat_samples = np.where(self.log, np.log10(flat_samples), flat_samples)
        # Add "log" to labels where needed
        labels = []
        theta_median = self.get_theta(which="median")
        for i, l in enumerate(self.labels):
            if self.log[i]:
                labels.append("log "+l)
                theta_median[i] = np.log10(theta_median[i])
            else:
                labels.append(l)
        labels = cornerhex_kwargs.pop("labels", labels)
        highlight = cornerhex_kwargs.pop("highlight", theta_median)
        return cornerhex.cornerplot(
            flat_samples,
            labels=labels,
            highlight=highlight,
            **cornerhex_kwargs,
        )
    
    
    def get_theta(self, which="median", discard=None, thin=None):
        """
        Function returns best or median parameter value.
        
        Parameters
        ----------
        which : str, optional, default: "median"
            Either "best" or "median".
            If "best" returns the parameters state vector with
            the highest log probability.
            If "median" returns the parameters state vector at the
            50 percentile.
        discard : int, optional, default: None
            Discard the first discard steps when picking samples.
            Defaults to three times the autocorrelation time.
        thin : int, optional, default: None
            Thin samples by thin when picking samples. Defaults
            to half of the autocorrelation time.
        
        Returns
        -------
        theta : array-like, (ndims, )
            Parameters state vector
        """
        if which=="best":
            # Get all chains and convert to parameter space
            samples = self.get_chain()
            samples = np.where(self.log[None, None, :], 10.**samples, samples)
            # Get log probability
            log_prob = self.get_log_prob()
            # Extrain chain and walker with highest log probability
            chain, walker = np.unravel_index(log_prob.argmax(), log_prob.shape)
            # Return state vector with highest log probability
            return samples[chain, walker, :]
        elif which=="median":
            # Get flat samples
            flat_samples = self.get_flat_samples(discard=discard, thin=thin)
            #Return state vector at 50 percentile.
            return np.percentile(flat_samples, 50, axis=0)
        else:
            raise RuntimeError("Unknown which: {}.".format(which))