import time
import numpy as np

import jax
import jax.numpy as jnp
import tracemalloc

import tinygp
import smolgp

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', family='sans serif', size=16)

key = jax.random.PRNGKey(0)

## Colors for benchmarking plots
colors = {'SSM':'#1f77b4',
          'QSM':'#ff7f0e', 
          'GP' :'#2ca02c',
          'pSSM':"#6A0E95"
        }

def generate_data(N, kernel, yerr=0.3):
    t_train = jnp.linspace(0, 86400, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true  = true_gp.sample(key=jax.random.PRNGKey(32)) 
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train

def default_block(y):
    return jnp.array(y).block_until_ready() 

def save_benchmark_data(filename, Ns, runtime_llh, memory_llh, outputs):
    import pickle
    data = {
        'Ns': Ns,
        'runtime_llh': runtime_llh,
        'memory_llh': memory_llh,
        'outputs': outputs,
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_benchmark_data(filename):
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['Ns'], data['runtime_llh'], data['memory_llh'], data['outputs']

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

def plot_benchmark(Ns, runtimes, ax=None, savefig=None, scale=True, powers={'SSM':1, 'QSM':1, 'GP':3, 'pSSM':1}):
    """
    runtimes should be a dict with values (mean, std) for each N
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,6), sharex=True)

    for name in runtimes:
        runtime_array = jnp.array(runtimes[name])
        mean_runtime = runtime_array[:,0]
        std_runtime  = runtime_array[:,1]
        if scale:
            scaled = scale_nans(runtime_array, Ns, power=powers[name])
            ax.errorbar(Ns, scaled[:,0], scaled[:,1], c=colors[name], ls=':')
        ax.errorbar(Ns, mean_runtime, std_runtime, c=colors[name], fmt='.-', label=name)

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


def benchmark_func(func, x, n_repeat=10, block_until_ready=True, block_output=default_block, trace_memory=False, **kwargs):
    """
    Benchmark the runtime of a function `func` with input `x`.
    """
    # Warm up (JIT compilation)
    if block_until_ready:
        out = func(x, **kwargs)
        block_output(out)
    # Timing & memory tracing
    runtimes = []
    mem_usages = []
    for _ in range(n_repeat):
        if trace_memory:
            tracemalloc.start()
        start = time.perf_counter()
        y = func(x, **kwargs)
        if block_until_ready:
            block_output(y) # ensure computation finished
        end = time.perf_counter()
        runtimes.append(end - start)
        if trace_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_usages.append(peak)

    if trace_memory:
        return (np.mean(runtimes), np.std(runtimes)), (np.mean(mem_usages), np.std(mem_usages)), y
    else:
        return np.mean(runtimes), np.std(runtimes), y

def benchmark(funcs, data, n_repeat=3, cutoffs={}, trace_memory=True, block=default_block):
    """
    Given some (jitted) functions, benchmark their runtimes over a range of input sizes.

    Parameters
    ----------
    funcs : list of callables
        List of functions to benchmark. Each function should take a single input array.
    data : list of tuples
        List of data tuples (t_train, y_train, yerr) for each input size

    Returns
    -------
    Ns : list of input sizes
    runtime : dict
        Dictionary mapping function names to lists of runtimes (means and stds) for each input size.
    memory : dict
        Dictionary mapping function names to lists of memory usages (means and stds) for each input size.
    outputs : dict
        Dictionary mapping function names to lists of outputs for each input size.
    """

    runtime = {}
    memory  = {}
    outputs = {}
    Ns = []
    for n in range(len(data)):
        N = data[n][0].shape[0]
        Ns.append(N)
        print(f'  ({n+1}/{len(data)}):  N = {N}')
        for name in funcs:
            func = funcs[name]
            cutoff = cutoffs.get(name, 3e4)

            if N <= cutoff:
                time, mem, val = benchmark_func(func, data[n], n_repeat=n_repeat, trace_memory=trace_memory, block_output=block)
                basestr = f'    {name}: time = {time[0]:.4f} ± {time[1]:.4f} s'
                memstr = f', mem = {format_bytes(mem[0])} ± {format_bytes(mem[1])}' if trace_memory else ''
                print(basestr + memstr)
            else:
                time, mem, val = (jnp.nan, jnp.nan), (jnp.nan, jnp.nan), jnp.nan
                print(f'    {name}: Skipped (N={N} > cutoff={cutoff})')
                
            if name not in runtime:
                runtime[name] = []
                memory[name]  = []
                outputs[name] = []

            runtime[name].append(time)
            memory[name].append(mem)
            outputs[name].append(val)

    return Ns, runtime, memory, outputs


def run_benchmark(true_kernel, funcs, yerr=0.3,
                  N_N=10, logN_min=1, logN_max=7, n_repeat=3, block=default_block,
                  cutoffs={}, trace_memory=True):
    """
    Generate data and benchmark the provided functions over a range of input sizes.
    """
    print('Generating data for benchmarking...')
    ## Generate all data ahead of time
    Ns = jnp.logspace(logN_min, logN_max, N_N).astype(int)
    data = []
    for N in Ns:
        t_train, y_train = generate_data(N, true_kernel, yerr=yerr)
        data.append(jnp.array([t_train, y_train, jnp.full_like(t_train, yerr)]))

    print('Running benchmark...')
    Ns, runtime, memory, outputs = benchmark(funcs, data, n_repeat=n_repeat, cutoffs=cutoffs, trace_memory=trace_memory, block=block)
    return Ns, runtime, memory, outputs


#################### LIKELIHOOD CONDITION ####################
def benchmark_llh(ssm_kernel, qsm_kernel, gp_kernel=None, true_kernel=None, yerr=0.3, **kwargs):
        
    block = default_block
    
    @jax.jit
    def ss_llh(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_ss = smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2)
        return gp_ss.log_probability(y_train)
    
    @jax.jit
    def qs_llh(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_qs = tinygp.GaussianProcess(qsm_kernel, t_train, diag=yerr**2)
        return gp_qs.log_probability(y_train)
    
    @jax.jit
    def gp_llh(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_gp = tinygp.GaussianProcess(gp_kernel, t_train, diag=yerr**2)
        return gp_gp.log_probability(y_train)

    @jax.jit
    def pss_llh(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_ss = smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2,
                                       solver=smolgp.solvers.ParallelStateSpaceSolver)
        return gp_ss.log_probability(y_train)
    
    funcs = {'SSM': ss_llh, 'QSM': qs_llh, 'GP': gp_llh, 'pSSM': pss_llh}
    
    true_kernel = qsm_kernel if true_kernel is None else true_kernel
    return run_benchmark(true_kernel, funcs, yerr=yerr, block=block, **kwargs)

#################### CONDITION BENCHMARK ####################
def benchmark_condition(ssm_kernel, qsm_kernel, gp_kernel=None, true_kernel=None, yerr=0.3, **kwargs):
    # def block(out):
    #     llh, condGP = out
    #     m=condGP.loc.block_until_ready() 
    #     P=condGP.variance.block_until_ready()
    #     # l=llh.block_until_ready()
    #     return m, P
    def block(out):
        m, P = out
        m.block_until_ready() 
        P.block_until_ready()
        return m, P

    @jax.jit
    def ss_cond(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_ss =smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2)
        llh, condGP_ss = gp_ss.condition(y_train)
        return condGP_ss.loc, condGP_ss.variance
    
    @jax.jit
    def qs_cond(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_qs = tinygp.GaussianProcess(qsm_kernel, t_train, diag=yerr**2)
        llh, condGP_qs = gp_qs.condition(y_train)
        return condGP_qs.loc, condGP_qs.variance
    
    @jax.jit
    def gp_cond(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_gp = tinygp.GaussianProcess(gp_kernel, t_train, diag=yerr**2)
        llh, condGP_gp = gp_gp.condition(y_train)
        return condGP_gp.loc, condGP_gp.variance    
    
    @jax.jit
    def pss_cond(data):
        t_train = data[0,:]
        y_train = data[1,:]
        yerr    = data[2,:]
        gp_ss = smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2,
                                       solver=smolgp.solvers.ParallelStateSpaceSolver)
        llh, condGP_ss = gp_ss.condition(y_train)
        return condGP_ss.loc, condGP_ss.variance    
    
    funcs = {'SSM': ss_cond, 'QSM': qs_cond, 'GP': gp_cond, 'pSSM': pss_cond}
        
    true_kernel = qsm_kernel if true_kernel is None else true_kernel
    return run_benchmark(true_kernel, funcs, yerr=yerr, block=block, **kwargs)


#################### PREDICT CONDITION ####################
def benchmark_prediction(ssm_kernel, qsm_kernel, gp_kernel=None, true_kernel=None, yerr=0.3,
                  N_M=10, logM_min=1, logM_max=7, n_repeat=3,
                  cutoffs={}, N=1000, trace_memory=True):
    
    runtime = {}
    memory  = {}
    outputs = {}

    def block(out):
        mu, var = out
        m=mu.block_until_ready() 
        P=var.block_until_ready()
        return m, P

    kernel = qsm_kernel if true_kernel is None else true_kernel
    Ms = jnp.logspace(logM_min, logM_max, N_M).astype(int)
    
    for M in Ms:
        # t_train, y_train = generate_data(N, yerr, baseline_minutes=900)
        t_train, y_train = generate_data(N, kernel)
        # t_train, y_train = generate_data(M, kernel)
        dt = 0.1 * (t_train.max()-t_train.min())
        t_test = jnp.linspace(t_train.min()-dt, t_train.max()+dt, M)

        ## Prepare GP objects
        gp_qs = tinygp.GaussianProcess(qsm_kernel, t_train, diag=yerr**2)
        gp_gp = tinygp.GaussianProcess(gp_kernel,  t_train, diag=yerr**2)
        gp_ss = smolgp.GaussianProcess(ssm_kernel, t_train, diag=yerr**2)
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
        
        funcs = {'SSM': ss_pred, 'QSM': qs_pred, 'GP': gp_pred}
        
        _, time, mem, val = benchmark(funcs, [t_test], n_repeat=n_repeat, 
                                    cutoffs=cutoffs, trace_memory=trace_memory, block=block)
        
        for name in funcs:
            if name not in runtime:
                runtime[name] = []
                memory[name]  = []
                outputs[name] = []

            runtime[name].append(time)
            memory[name].append(mem)
            outputs[name].append(val)
            
    return Ms, runtime, memory, outputs

def format_bytes(n):
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"