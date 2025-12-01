import jax
import jax.numpy as jnp
import tinygp

key = jax.random.PRNGKey(0)


def allclose(name, residuals, tol, atol=1e-14):
    """
    Check all residuals are < tol
    if they are, but aren't < atol, print a warning
    """
    maxres = jnp.max(jnp.abs(residuals))
    assert maxres < tol, (
        f"{name} did not agree to within desired tolerance."
        f" Maximum absolute deviation is {maxres:.3e} "
    )
    if maxres < atol:
        print(f"    ...{name}: agrees exactly (<{maxres:.0e})")
    else:
        print(f"    ...{name}: agrees (WARNING: only to < {maxres:.1e})")


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


def generate_data(N, kernel, yerr=0.3, tmin=0, tmax=86400):
    t_train = jnp.linspace(tmin, tmax, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train)
    y_true = true_gp.sample(key=key)
    y_train = y_true + yerr * jax.random.normal(key, shape=(N,))
    return t_train, y_train


def generate_integrated_data(N, kernel, texp=180, yerr=0.3, readout=40):
    # Generate true GP over baseline
    cadence = texp + readout
    tmin = 0
    tmax = N * cadence
    buffer = 0.1 * (tmax - tmin)
    t = jnp.arange(tmin - buffer, tmax + buffer, 1)
    true_gp = tinygp.GaussianProcess(kernel, t)
    y = true_gp.sample(key=key)

    # Generate synthetic observations
    @jax.jit
    def make_exposure(tmid, texp):
        t_in_exp = jnp.linspace(tmid - texp / 2, tmid + texp / 2, 50)
        # quickly slice region of interest
        idx = jnp.searchsorted(t, t_in_exp, side="right")
        idx = jnp.clip(idx, 0, t.size - 2)
        # just interpolate that region
        y_in_exp = jnp.interp(t_in_exp, t[idx], y[idx])
        return jnp.mean(y_in_exp)

    texp_train = jnp.full(N, texp)  # constant exposure time
    t_train = jnp.arange(tmin, tmax, cadence)
    y_true = jax.vmap(make_exposure)(t_train, texp_train)
    y_train = y_true + yerr * jax.random.normal(key, shape=(len(t_train),))
    return t_train, y_train


def get_data(true_kernel, N, yerr=0.3, exposure_quantities=None, save=True):
    # Generate data of length N
    if exposure_quantities:
        texp, readout = exposure_quantities
        t_train, y_train = generate_integrated_data(
            N, true_kernel, texp=texp, readout=readout, yerr=yerr
        )
        texp_train = jnp.full_like(t_train, texp)
        yerr_train = jnp.full_like(t_train, yerr)
        instid = jnp.full_like(t_train, 0)
        data = jnp.array([t_train, y_train, yerr_train, texp_train, instid])
        savename = f"data/{N}_int.npz"
    else:
        t_train, y_train = generate_data(N, true_kernel, yerr=yerr)
        yerr_train = jnp.full_like(t_train, yerr)
        data = jnp.array([t_train, y_train, yerr_train])
        savename = f"data/{N}.npz"
    if save:
        jnp.savez(savename, data)
    return data
