import time
import numpy as np

import jax
import jax.numpy as jnp

import tinygp
import smolgp

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', family='sans serif', size=16)

key = jax.random.PRNGKey(0)

## Colors for benchmarking plots
cSSM = '#1f77b4'
cQSM = '#ff7f0e'
cGP  = '#2ca02c'

def generate_data(N, kernel, yerr=0.3):
    t_train = jnp.linspace(0, 86400, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true  = true_gp.sample(key=jax.random.PRNGKey(32)) 
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train

def default_block(y):
    return jnp.array(y).block_until_ready() 

def benchmark(func, x, n_repeat=10, block_until_ready=True, block_output=default_block, **kwargs):
    # Warm up (JIT compilation)
    if block_until_ready:
        out = func(x, **kwargs)
        block_output(out)
    runtimes = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        y = func(x, **kwargs)
        if block_until_ready:
            block_output(y) # ensure computation finished
        end = time.perf_counter()
        runtimes.append(end - start)
    return np.mean(runtimes), np.std(runtimes), y

def scale_nans(runtime_array, Ns, power=1):
    # Find the last non-nan index
    last_valid_idx = jnp.where(~jnp.isnan(runtime_array[:, 0]))[0][-1]
    last_valid_runtime = runtime_array[last_valid_idx, 0]
    last_valid_N = Ns[last_valid_idx]

    # Scale the nans based on the last valid point
    for i in range(last_valid_idx + 1, len(runtime_array)):
        runtime_array = runtime_array.at[i, 0].set(last_valid_runtime * (Ns[i] / last_valid_N)**power)
        runtime_array = runtime_array.at[i, 1].set(0)  # Set std to 0 for scaled points

    return runtime_array

def plot_benchmark(Ns, runtime_ss, runtime_qs, runtime_gp, ax=None,
                   labels=['SSM', 'QSM', 'GP'], savefig=None, powers=[1,1,3]):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,6), sharex=True)

    scaled_ss = scale_nans(runtime_ss, Ns, power=powers[0])
    scaled_qs = scale_nans(runtime_qs, Ns, power=powers[1])
    scaled_gp = scale_nans(runtime_gp, Ns, power=powers[2])
    ax.errorbar(Ns, scaled_ss[:,0], scaled_ss[:,1], c=cSSM, ls='--')
    ax.errorbar(Ns, scaled_qs[:,0], scaled_qs[:,1], c=cQSM, ls='--')
    ax.errorbar(Ns, scaled_gp[:,0], scaled_gp[:,1], c=cGP,  ls='--')

    ax.errorbar(Ns, runtime_ss[:,0], runtime_ss[:,1], c=cSSM, fmt='.-', label=labels[0])
    ax.errorbar(Ns, runtime_qs[:,0], runtime_qs[:,1], c=cQSM, fmt='.-', label=labels[1])
    ax.errorbar(Ns, runtime_gp[:,0], runtime_gp[:,1], c=cGP,  fmt='.-', label=labels[2])

    ax.legend();
    ax.set(xscale='log', yscale='log', xlabel='Number of data points', ylabel='Runtime [s]');
    tens = 10**jnp.arange(jnp.log10(Ns[0]), jnp.log10(Ns[-1])+1)
    ax.set_xticks(tens, labels=['' for _ in tens], minor=True)
    ax.grid(alpha=0.5, zorder=-10);
    ax.grid(alpha=0.5, zorder=-10, which='minor', axis='x');
    # ax.set_ylim(top=jnp.nanmax(runtime_ss[:,0])*1e3)
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    return ax


def benchmark_llh(ssSHO, qsSHO, gpSHO=None, true_kernel=None, yerr=0.3,
                  N_N=10, logN_min=1, logN_max=7, n_repeat=3,
                  ss_cutoff=1e6, gp_cutoff=1e4, qs_cutoff=1e6):
    
    kernel = qsSHO if true_kernel is None else true_kernel
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    runtime_ss = []
    runtime_qs = []
    runtime_gp = []
    outputs = []
    for N in Ns:
        # t_train, y_train = generate_data(N, yerr, baseline_minutes=900)
        t_train, y_train = generate_data(N, kernel)

        @jax.jit
        def ss_llh(y_train):
            gp_ss =smolgp.GaussianProcess(ssSHO, t_train, diag=yerr**2)
            return gp_ss.log_probability(y_train)
        
        @jax.jit
        def qs_llh(y_train):
            gp_qs = tinygp.GaussianProcess(qsSHO, t_train, diag=yerr**2)
            return gp_qs.log_probability(y_train)
        
        @jax.jit
        def gp_llh(y_train):
            gp_gp = tinygp.GaussianProcess(gpSHO, t_train, diag=yerr**2)
            return gp_gp.log_probability(y_train)
        
        ## Quasiseparable GP
        if N <= qs_cutoff:
            mean_qs, std_qs, val_qs = benchmark(qs_llh, y_train, n_repeat=n_repeat)
        else:
            mean_qs, std_qs, val_qs = np.nan, np.nan, np.nan
        
        ## State space GP
        if N <= ss_cutoff:
            mean_ss, std_ss, val_ss = benchmark(ss_llh, y_train, n_repeat=n_repeat)
        else:
            mean_ss, std_ss, val_ss = np.nan, np.nan, np.nan
        
        ## Full (dense) GP, if provided
        if gpSHO is not None:
            if N <= gp_cutoff:
                mean_gp, std_gp, val_gp = benchmark(gp_llh, y_train, n_repeat=n_repeat)
            else:
                mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan
        else:
            mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan

        runtime_ss.append([mean_ss, std_ss])
        runtime_qs.append([mean_qs, std_qs])
        runtime_gp.append([mean_gp, std_gp])
        outputs.append([val_ss, val_qs, val_gp])

    outputs    = jnp.array(outputs)
    runtime_ss = jnp.array(runtime_ss)
    runtime_qs = jnp.array(runtime_qs)
    runtime_gp = jnp.array(runtime_gp)
    return (Ns, outputs), (runtime_ss, runtime_qs, runtime_gp)



def benchmark_condition(ssSHO, qsSHO, gpSHO=None, true_kernel=None, yerr=0.3,
                  N_N=10, logN_min=1, logN_max=7, n_repeat=3,
                  ss_cutoff=1e6, gp_cutoff=1e4, qs_cutoff=1e6):
    
    def block(out):
        llh, condGP = out
        m=condGP.loc.block_until_ready() 
        P=condGP.variance.block_until_ready()
        # l=llh.block_until_ready()
        return m, P

    kernel = qsSHO if true_kernel is None else true_kernel
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    runtime_ss = []
    runtime_qs = []
    runtime_gp = []
    outputs = []
    for N in Ns:
        # t_train, y_train = generate_data(N, yerr, baseline_minutes=900)
        t_train, y_train = generate_data(N, kernel)

        @jax.jit
        def ss_cond(y_train):
            gp_ss =smolgp.GaussianProcess(ssSHO, t_train, diag=yerr**2)
            return gp_ss.condition(y_train)
        
        @jax.jit
        def qs_cond(y_train):
            gp_qs = tinygp.GaussianProcess(qsSHO, t_train, diag=yerr**2)
            return gp_qs.condition(y_train)
        
        @jax.jit
        def gp_cond(y_train):
            gp_gp = tinygp.GaussianProcess(gpSHO, t_train, diag=yerr**2)
            return gp_gp.condition(y_train)
        
        ## Quasiseparable GP
        if N <= qs_cutoff:
            mean_qs, std_qs, val_qs = benchmark(qs_cond, y_train, block_output=block, n_repeat=n_repeat)
        else:
            mean_qs, std_qs, val_qs = np.nan, np.nan, np.nan
        
        ## State space GP
        if N <= ss_cutoff:
            mean_ss, std_ss, val_ss = benchmark(ss_cond, y_train, block_output=block, n_repeat=n_repeat)
        else:
            mean_ss, std_ss, val_ss = np.nan, np.nan, np.nan
        
        ## Full (dense) GP, if provided
        if gpSHO is not None:
            if N <= gp_cutoff:
                mean_gp, std_gp, val_gp = benchmark(gp_cond, y_train, block_output=block, n_repeat=n_repeat)
            else:
                mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan
        else:
            mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan

        runtime_ss.append([mean_ss, std_ss])
        runtime_qs.append([mean_qs, std_qs])
        runtime_gp.append([mean_gp, std_gp])
        outputs.append([val_ss, val_qs, val_gp])

    # outputs    = jnp.array(outputs)
    runtime_ss = jnp.array(runtime_ss)
    runtime_qs = jnp.array(runtime_qs)
    runtime_gp = jnp.array(runtime_gp)
    return (Ns, outputs), (runtime_ss, runtime_qs, runtime_gp)



def benchmark_prediction(ssSHO, qsSHO, gpSHO=None, true_kernel=None, yerr=0.3,
                  N_M=10, logM_min=1, logM_max=7, n_repeat=3,
                  ss_cutoff=1e6, gp_cutoff=1e4, qs_cutoff=1e6,
                  N=1000):
    
    def block(out):
        mu, var = out
        m=mu.block_until_ready() 
        P=var.block_until_ready()
        return m, P

    kernel = qsSHO if true_kernel is None else true_kernel
    Ms = jnp.logspace(logM_min, logM_max, N_M).astype(int)
    runtime_ss = []
    runtime_qs = []
    runtime_gp = []
    outputs = []
    for M in Ms:
        # t_train, y_train = generate_data(N, yerr, baseline_minutes=900)
        t_train, y_train = generate_data(N, kernel)
        # t_train, y_train = generate_data(M, kernel)
        dt = 0.1 * (t_train.max()-t_train.min())
        t_test = jnp.linspace(t_train.min()-dt, t_train.max()+dt, M)

        ## Prepare GP objects
        gp_qs = tinygp.GaussianProcess(qsSHO, t_train, diag=yerr**2)
        gp_gp = tinygp.GaussianProcess(gpSHO, t_train, diag=yerr**2)
        gp_ss =smolgp.GaussianProcess(ssSHO, t_train, diag=yerr**2)
        _, condGPss = gp_ss.condition(y_train)
        
        ## Only time the actual prediction part
        @jax.jit
        def ss_pred(t_test):
            # _, condGPss = gp_ss.condition(y_train)
            return condGPss.predict(t_test, return_var=True)
        
        @jax.jit
        def qs_pred(t_test):
            return gp_qs.predict(y_train, t_test, return_var=True)
        
        @jax.jit
        def gp_pred(t_test):
            return gp_gp.predict(y_train, t_test, return_var=True)
        
        ## Quasiseparable GP
        if M <= qs_cutoff:
            mean_qs, std_qs, val_qs = benchmark(qs_pred, t_test, block_output=block, n_repeat=n_repeat)
        else:
            mean_qs, std_qs, val_qs = np.nan, np.nan, np.nan
        
        ## State space GP
        if M <= ss_cutoff:
            mean_ss, std_ss, val_ss = benchmark(ss_pred, t_test, block_output=block, n_repeat=n_repeat)
        else:
            mean_ss, std_ss, val_ss = np.nan, np.nan, np.nan
        
        ## Full (dense) GP, if provided
        if gpSHO is not None:
            if M <= gp_cutoff:
                mean_gp, std_gp, val_gp = benchmark(gp_pred, t_test, block_output=block, n_repeat=n_repeat)
            else:
                mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan
        else:
            mean_gp, std_gp, val_gp = np.nan, np.nan, np.nan

        runtime_ss.append([mean_ss, std_ss])
        runtime_qs.append([mean_qs, std_qs])
        runtime_gp.append([mean_gp, std_gp])
        outputs.append([val_ss, val_qs, val_gp])

    # outputs    = jnp.array(outputs)
    runtime_ss = jnp.array(runtime_ss)
    runtime_qs = jnp.array(runtime_qs)
    runtime_gp = jnp.array(runtime_gp)
    return (Ms, outputs), (runtime_ss, runtime_qs, runtime_gp)
