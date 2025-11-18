import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

## Colors for benchmarking plots
colors = {"SSM" : "#1f77b4",
          "QSM" : "#ff7f0e", 
          "GP"  : "#2ca02c", 
          "pSSM": "#6A0E95"
          }
markers = {"SSM" : "o",
           "QSM" : "s", 
           "GP"  : "D", 
           "pSSM": "*"
          }
markersize = {"SSM" : 8,
              "QSM" : 6, 
              "GP"  : 6, 
              "pSSM": 10,
          }

def symlogticks(vmin, vmax, linthresh=1e-16, spacing=1):
    """Generate ticks for a symlog axis."""
    thresh = int(jnp.log10(linthresh))
    negticks = -10.**np.arange(thresh, vmin, spacing)[::-1]
    posticks = 10.**jnp.arange(thresh, vmax, spacing)
    return jnp.concatenate([negticks, posticks])

def scale_nans(runtime_array, Ns, power=1):
    # Find the last non-nan index
    last_valid_idx = jnp.where(~jnp.isnan(runtime_array[:, 0]))[0][-1]
    last_valid_runtime = runtime_array[last_valid_idx, 0]
    last_valid_N = Ns[last_valid_idx]

    # Scale the nans based on the last valid point
    for i in range(last_valid_idx + 1, len(runtime_array)):
        runtime_array = runtime_array.at[i, 0].set(
            last_valid_runtime * (Ns[i] / last_valid_N) ** power
        )
        runtime_array = runtime_array.at[i, 1].set(0)  # Set std to 0 for scaled points

    return runtime_array

def plot_benchmark(
    Ns,
    runtimes,
    ax=None,
    savefig=None,
    scale=True,
    powers={"SSM": 1, "QSM": 1, "GP": 3, "pSSM": 1},
    labels=None,
):
    """
    runtimes should be a dict with values (mean, std) for each N
    """
    if labels is None:
        labels = {name: name for name in runtimes}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True)

    for name in runtimes:
        runtime_array = jnp.array(runtimes[name])
        mean_runtime = runtime_array[:, 0]
        std_runtime = runtime_array[:, 1]
        if scale:
            scaled = scale_nans(runtime_array, Ns, power=powers[name])
            ax.errorbar(Ns, scaled[:, 0], scaled[:, 1], c=colors[name], ls=":")
        ax.errorbar(Ns, mean_runtime, std_runtime, c=colors[name], marker=markers[name],
                     markersize=markersize[name], fmt='-', label=labels[name])

    ax.legend()
    ax.set(
        xscale="log", yscale="log", xlabel="Number of data points", ylabel="Runtime [s]"
    )
    tens = 10 ** jnp.arange(jnp.log10(Ns[0]), jnp.log10(Ns[-1]) + 1)
    ax.set_xticks(tens, labels=["" for _ in tens], minor=True)
    ax.grid(alpha=0.5, zorder=-10)
    ax.grid(alpha=0.5, zorder=-10, which="minor", axis="x")
    # ax.set_ylim(top=jnp.nanmax(runtime_ss[:,0])*1e3)
    
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    return ax