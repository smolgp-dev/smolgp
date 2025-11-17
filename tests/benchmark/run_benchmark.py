import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import tinygp
import smolgp
from benchmark import *

import sys
sys.path.insert(0, '/mnt/home/rrubenzahl/solar/onefit/onefit/')

import importlib
import gpkernels
importlib.reload(gpkernels)
from gpkernels import *

key = jax.random.PRNGKey(0)

##############
yerr = 0.3
##############

S=2.36
w=0.0195
Q=7.63
sigma = jnp.sqrt(S*w*Q)
qsSHO = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
ssSHO =smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
gpSHO = gpkernels.OscillationKernel()


which = sys.argv[1]

if which=='llh':
    print('Benchmarking likelihood...')
    llh_filename = 'results/llh_benchmark.pkl'
    Ns, runtime_llh, memory_llh, outputs = benchmark_llh(ssSHO, qsSHO, gp_kernel=gpSHO, 
                                                         true_kernel=qsSHO, yerr=yerr,
                                                         n_repeat=5, trace_memory=True,
                                                         N_N=13, logN_min=1, logN_max=7,
                                                         cutoffs={'GP':3e4, 'SSM':1e7, 'QSM':1e7, 'pSSM':1e7}
                                                         )
    save_benchmark_data(llh_filename, Ns, runtime_llh, memory_llh, outputs)
    print('Wrote results to', llh_filename)

elif which=='cond':
    print('Benchmarking condition...')
    cond_filename = 'results/cond_benchmark.pkl'
    Ns, runtime_cond, memory_cond, outputs = benchmark_condition(ssSHO, qsSHO, gp_kernel=gpSHO,
                                                                true_kernel=qsSHO, yerr=yerr,
                                                                N_N=10, n_repeat=5, 
                                                                logN_min=1, logN_max=7,
                                                                cutoffs={'GP':3e4, 'SSM':1e6, 'QSM':1e6, 'pSSM':1e6})
    
    save_benchmark_data(cond_filename, Ns, runtime_cond, memory_cond, outputs)
    print('Wrote results to', cond_filename)

else:
    raise ValueError("Argument must be 'llh' or 'cond'")