import jax
import jax.numpy as jnp
import tinygp

import smolgp

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
    """Draw ``N`` instantaneous measurements from the process defined by ``kernel``.

    ``kernel`` must be a tinygp.kernels.Kernel object. We use tinygp here over
    smolgp as it is ~10x faster at sampling from the prior for instantaneous data.

    Returns data on an evenly spaced grid between ``tmin`` and ``tmax``, sampled
    with measurement noise `yerr``.
    """
    t_train = jnp.linspace(tmin, tmax, N)
    true_gp = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    y_train = true_gp.sample(key=key)
    return t_train, y_train


def generate_integrated_data(N, kernel, texp=180, yerr=0.3, readout=40):
    """Draw ``N`` exposure-integrated observationsfrom the process defined by ``kernel``.

    ``kernel`` must be a smolgp :class:`~smolgp.kernels.integrated.IntegratedStateSpaceModel`, 
    e.g. ``smolgp.kernels.IntegratedSHO``. Draws exposure-averaged samples from
    the process, replacing the old method of sampling a much higher resolution 
    grid at instantaneous times and averaging within windows to create exposures,
    which is prohibitively expensive for large simulated datasets.

    The exposures are laid out back to back at ``texp + readout``, enusring
    ``texp < cadence`` so they never overlap and a single instrument index suffices.

    Args:
        N: number of exposures.
        kernel: integrated state-space kernel to draw from.
        texp: exposure duration.
        yerr: white measurement noise added to the draw.
        readout: dead time between exposures.

    Returns:
        ``(t_train, y_train)`` -- exposure midpoints and noisy integrated observations.
    """
    cadence = texp + readout
    # arange(N) * cadence rather than arange(0, N * cadence, cadence): the
    # latter can land on N +/- 1 elements depending on floating point.
    t_train = jnp.arange(N) * float(cadence)
    texp_train = jnp.full(N, float(texp))
    instid = jnp.zeros(N, dtype=int)

    # Sample with measurement noise
    gp = smolgp.GaussianProcess(kernel, X=(t_train, texp_train, instid), noise=yerr**2)
    y_train = gp.sample(key)
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
