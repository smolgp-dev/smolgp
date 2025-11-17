import logging
# Suppress only JAX XLA bridge warnings
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import tinygp
import smolgp
from benchmark import *

import sys
import benchmark.kernels as testgp

key = jax.random.PRNGKey(0)


#################### LIKELIHOOD FUNCTIONS FOR BENCHMARK ####################
def ss_llh(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2)
    return gp_ss.log_probability(y_train)

def qs_llh(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    return gp_qs.log_probability(y_train)

def gp_llh(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_gp = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    return gp_gp.log_probability(y_train)

def pss_llh(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2,
                                    solver=smolgp.solvers.ParallelStateSpaceSolver)
    return gp_ss.log_probability(y_train)

#################### CONDITION FUNCTIONS FOR BENCHMARK ####################

def ss_cond(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2)
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

def qs_cond(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    llh, condGP_qs = gp_qs.condition(y_train)
    return jnp.array([condGP_qs.loc, condGP_qs.variance])

def gp_cond(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_gp = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    llh, condGP_gp = gp_gp.condition(y_train)
    return jnp.array([condGP_gp.loc, condGP_gp.variance])

def pss_cond(data, kernel):
    t_train = data[0,:]
    y_train = data[1,:]
    yerr    = data[2,:]
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2,
                                    solver=smolgp.solvers.ParallelStateSpaceSolver)
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])    



######################################## MAIN ########################################
if __name__ == '__main__':

    yerr = 0.3
    which = sys.argv[1]

    S=2.36
    w=0.0195
    Q=7.63
    sigma = jnp.sqrt(S*w*Q)
    qsm_kernel = tinygp.kernels.quasisep.SHO(omega=w, quality=Q, sigma=sigma)
    ssm_kernel = smolgp.kernels.SHO(omega=w, quality=Q, sigma=sigma)
    gp_kernel  = testgp.SHOKernel(w=w, Q=Q, S=S)
    true_kernel = qsm_kernel

    kernels = {'SSM': ssm_kernel, 'QSM': qsm_kernel, 'GP': gp_kernel, 'pSSM': ssm_kernel}

    if which=='llh':
        print('Benchmarking likelihood...')
        out_filename = 'results/llh_benchmark.pkl'
        funcs = {'SSM': ss_llh, 'QSM': qs_llh, 'GP': gp_llh, 'pSSM': pss_llh}    
        Ns, runtime, memory, outputs = run_benchmark(true_kernel, funcs, kernels, yerr=yerr,
                                                     n_repeat=5, N_N=20, logN_min=1, logN_max=7,
                                                     cutoffs={'GP':6e4, 'SSM':1e7, 'QSM':1e7, 'pSSM':1e7}
                                                     )
    elif which=='cond':
        print('Benchmarking condition...')
        out_filename = 'results/cond_benchmark.pkl'
        funcs = {'SSM': ss_cond, 'QSM': qs_cond, 'GP': gp_cond, 'pSSM': pss_cond}    
        Ns, runtime, memory, outputs = run_benchmark(true_kernel, funcs, kernels, yerr=yerr,
                                                     n_repeat=5, N_N=10, logN_min=1, logN_max=7,
                                                     cutoffs={'GP':3e4, 'SSM':1e6, 'QSM':1e6, 'pSSM':1e6}
                                                     )
    else:
        raise ValueError("Argument must be 'llh' or 'cond'")

    print('Wrote results to', out_filename)
    save_benchmark_data(out_filename, Ns, runtime, memory, outputs)
