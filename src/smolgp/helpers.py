import heapq

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from tinygp.helpers import JAXArray


def assign_min_instids(t: JAXArray, delta: JAXArray) -> tuple[JAXArray, int]:
    r"""Compute the minimum number of non-overlapping ``instid`` groups for a
    set of ``M`` exposure windows :math:`(t_i - \delta_i/2,\; t_i + \delta_i/2)`
    with arbitrary overlap.

    Main use is in :func:`~smolgp.solvers.sample.merge_exposure_test_coords`,
    to optimally reduce the dimensionality of the augmented state. Because
    sample draws are from a joint multivariate Gaussian, they cannot be
    drawn independently for overlapping exposures, so we must track them
    with separate instrument indices. The sampling algorithm scales with
    O(n^3), n the number of instruments, so we want to minimize that number.

    This is the "minimum number of meeting rooms" problem, which is optimally
    solved with the standard "reuse whichever group's window finished earliest,
    if it's  already finished" greedy sweep, via a min-heap keyed by end time.
    Cost is :math:`O(M \log M)`.

    Written in plain Python/``heapq`` as it is a one-time preprocessing step
    and defines the shape of the augmented state, two things that are not jittable.

    Args:
        t: Exposure midpoints, length ``M``.
        delta: Exposure widths, length ``M`` (must be >= 0).

    Returns:
        instid: Length ``M`` integer array of group assignments,
            ``0 <= instid[i] < num_insts``.
        num_insts: The minimum number of "instruments" i.e. ``int(jnp.max(instid)) + 1``.
    """
    a = [float(x) for x in (t - delta / 2)]
    b = [float(x) for x in (t + delta / 2)]
    M = len(a)
    order = sorted(range(M), key=lambda i: a[i])

    heap: list[tuple[float, int]] = []  # (end_time, group id), min-heap by end_time
    instid = [0] * M
    next_id = 0
    for i in order:
        if heap and heap[0][0] <= a[i]:
            _end_time, gid = heapq.heappop(heap)
        else:
            gid = next_id
            next_id += 1
        instid[i] = gid
        heapq.heappush(heap, (b[i], gid))

    return jnp.array(instid, dtype=int), next_id


def check_no_overlap_within_instid(
    t: JAXArray, delta: JAXArray, instid: JAXArray
) -> None:
    r"""Raise if any two exposure windows sharing an ``instid`` overlap.

    One physical instrument cannot be mid-exposure twice at once, so this is
    already an implicit fact about real data. It becomes a *checkable*
    precondition when a user supplies ``instid`` for exposure-integrated
    **test** points (see :meth:`smolgp.gp.GaussianProcess.sample`), where the
    id both selects the observation model and picks which augmented "probe"
    dimension the window accumulates into. Two overlapping windows sharing a
    dimension would have the second window's reset silently clobber the
    first's in-progress integral, so this fails loudly instead.

    Uses the same convention as :func:`assign_min_instids`: windows that
    merely touch (one ends exactly where the next begins) do **not** overlap.

    Zero-width (``delta == 0``, i.e. instantaneous) points are exempt and
    ignored entirely: they are read out through the ordinary observation
    model, and their reset is redirected to a shared "trash" dimension by
    :func:`~smolgp.solvers.sample.merge_exposure_test_coords`, so they can
    neither clobber nor be clobbered by a genuine exposure sharing their id.

    Args:
        t: Exposure midpoints, length ``M``.
        delta: Exposure widths, length ``M``.
        instid: Instrument id per exposure, length ``M``.

    Raises:
        ValueError: if two windows sharing an ``instid`` overlap, naming the
            offending pair.
    """
    a = [float(x) for x in (t - delta / 2)]
    b = [float(x) for x in (t + delta / 2)]
    ids = [int(x) for x in instid]
    widths = [float(x) for x in delta]

    by_inst: dict[int, list[int]] = {}
    for i, g in enumerate(ids):
        if widths[i] > 0:  # delta == 0 points are exempt (see above)
            by_inst.setdefault(g, []).append(i)

    for g, idxs in sorted(by_inst.items()):
        # Within one instrument, sort by start time; only adjacent pairs can
        # be the first overlap, so one linear sweep suffices.
        idxs.sort(key=lambda i: a[i])
        for prev, cur in zip(idxs, idxs[1:]):
            if a[cur] < b[prev]:
                raise ValueError(
                    f"Exposure windows {prev} and {cur} both have instid={g} but "
                    f"overlap: [{a[prev]:g}, {b[prev]:g}] and [{a[cur]:g}, {b[cur]:g}]. "
                    "One instrument cannot record two overlapping exposures. Give "
                    "them distinct instid values, or omit instid entirely (pass "
                    "X_test=(t, delta)) to have non-overlapping groups assigned "
                    "automatically."
                )


def block_view(A, b):
    Nb, Mb = A.shape
    assert Nb % b == 0 and Mb % b == 0
    N = Nb // b
    M = Mb // b
    return A.reshape(N, b, M, b).transpose(0, 2, 1, 3)


def Q_from_VanLoan(F: JAXArray, L: JAXArray, Qc: JAXArray, dt: JAXArray) -> JAXArray:
    r"""Compute the process noise covariance via the Van Loan method.

    Evaluates

    .. math::

        Q_k = \int_0^{\Delta t} e^{F(\Delta t - s)}\, L\, Q_c\, L^T\, e^{F^T(\Delta t - s)}\, ds

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F` from :meth:`~smolgp.kernels.StateSpaceModel.design_matrix`.
        L: Noise effect matrix :math:`L` from :meth:`~smolgp.kernels.StateSpaceModel.noise_effect_matrix`.
        Qc: Spectral density :math:`Q_c` from :meth:`~smolgp.kernels.StateSpaceModel.noise`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Process noise covariance matrix :math:`Q_k` over time step :math:`\Delta t`.
    """
    QL = L @ Qc @ L.T
    b = len(F)  # block size
    Z = jnp.zeros_like(F)
    C = jnp.block([[-F, QL], [Z, F.T]])
    VanLoanBlock = expm(C * dt)
    G2 = VanLoanBlock[:b, b:]
    F3 = VanLoanBlock[b:, b:]
    return F3.T @ G2


def Phibar_from_VanLoan(F: JAXArray, dt: JAXArray) -> JAXArray:
    r"""Compute the integrated transition matrix via the Van Loan method.

    Evaluates

    .. math::

        \bar{\Phi} = \int_0^{\Delta t} e^{F s}\, ds

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F` from :meth:`~smolgp.kernels.StateSpaceModel.design_matrix`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Integrated transition matrix :math:`\bar{\Phi}` over time step :math:`\Delta t`.
    """
    b = len(F)  # block size
    Z = jnp.zeros((b, b))
    I = jnp.eye(b)
    C = jnp.block([[F, I], [Z, Z]])
    VanLoanBlock = expm(C * dt)
    G3 = VanLoanBlock[:b, b:]
    return G3


def robust_sqrt(M: JAXArray) -> JAXArray:
    r"""Symmetric-PSD matrix square root via eigendecomposition.

    Returns :math:`S` such that :math:`S S^T = M`, via
    :math:`M = V \mathrm{diag}(w) V^T`, :math:`S = V \mathrm{diag}(\sqrt{\max(w,0)})`.

    In cases where :math:`M` is numerically singular, :func:`jnp.linalg.cholesky` will fail.
    This method, while slightly slower, is robust to singularity and returns a valid square root.
    Needed for sampling (which uses a square root of the covariance) when either
    1. Q_k is exactly zero, either from a zero-length transition (such as the first step,
       or two states at the same instant), or for kernels with Q_k=0 everwhere (e.g. Cosine)
    2. Q_k is numerically singular due to multiple instruments resetting at the same instant,
       which produces perfectly correlated integral states (see docstring of :func:`get_smoothing_gain`).
    """
    w, V = jnp.linalg.eigh(M)
    return V * jnp.sqrt(jnp.clip(w, min=0.0))[None, :]


def get_smoothing_gain(P_pred_next: JAXArray, numerator: JAXArray) -> JAXArray:
    r"""Computes the RTS smoothing gain :math:`G_k` from

    .. math::

        G_k^T = \mathrm{solve}(\mathbf{P}_{\mathrm{pred,next}}^T,\ \mathrm{numerator}^T),

    guarding against :math:`\mathbf{P}_{\mathrm{pred,next}}` being exactly singular. This
    arises when two states in an exposure-aware model occupy the same instant in time,
    which produces a singular covariance in two different ways:
        1. An exposure-start reset zeroes a row/column of the covariance. If the following
           transition has zero duration, that zeroed row/column passes through unregularized.
        2. At *nonzero* transition lengths, whenever two or more instruments are reset
           at the exact same instant, their integral states become perfectly (diagonal and
           row identical) correlated, since they share the same driving process noise.
           This can persist through several further transitions before it clears.

    Checking ``Delta == 0`` only catches the first case, so we instead directly check for
    singularity in :math:`\mathbf{P}_{\mathrm{pred,next}}` itself.

    The detection is via ``P_pred_next``'s (scale-normalized) log-determinant: dividing
    by ``trace(P_pred_next) / n`` before taking ``slogdet`` makes the threshold
    independent of the kernel's overall amplitude (a plain absolute threshold on the raw
    determinant would not be, since determinant scales as amplitude\ :sup:`n`). A
    genuinely singular matrix here shows up many orders of magnitude below this
    threshold; typically ``-inf`` to around ``-40``, whereas well-conditioned states
    have around ``-2`` to ``-24``. The threshold set here is ``-30``.

    Both branches compute the correct smoothing gain by inverting the predicted covariance, but with different methods:
    - The common (non-singular) case uses :func:`jnp.linalg.solve` (LU-based, cheap), which assumes invertibility.
    - The degenerate case uses :func:`jnp.linalg.lstsq` (SVD-based), which is more expensive but handles singular matrices correctly.

    TLDR; usage converts
        ``G_k = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T``
    to
        ``G_k = get_smoothing_gain(P_pred_next, P_k @ A_k.T)``
    """
    n = P_pred_next.shape[0]
    scale = jnp.trace(P_pred_next) / n
    sign, logdet = jnp.linalg.slogdet(P_pred_next)
    logdet_normalized = logdet - n * jnp.log(scale)
    is_singular = (sign <= 0) | (logdet_normalized < -30.0)

    def solve_generic(_):
        return jnp.linalg.solve(P_pred_next.T, numerator.T).T

    def solve_degenerate(_):
        Y, *_ = jnp.linalg.lstsq(P_pred_next.T, numerator.T)
        return Y.T

    return jax.lax.cond(is_singular, solve_degenerate, solve_generic, operand=None)


def VanLoan(
    F: JAXArray, L: JAXArray, Qc: JAXArray, dt: JAXArray
) -> dict[str, JAXArray]:
    r"""Compute all submatrices of the Van Loan matrix exponential.

    Assembles the block matrix :math:`C` and returns its matrix exponential,
    partitioned into the submatrices ``F1``-``F4``, ``G1``-``G3``, ``H1``-``H2``,
    ``K1`` (see Van Loan 1978 for notation), from which various integrals such as
    :func:`Q_from_VanLoan` and :func:`Phibar_from_VanLoan` can be derived.

    See `Van Loan (1978) <https://ecommons.cornell.edu/items/cba38b2e-6ad4-45e6-8109-0a019fe5114c>`_,
    "Computing Integrals Involving the Matrix Exponential" (`PDF <https://www.olemartin.no/artikler/vanloan.pdf>`_).

    Args:
        F: Feedback (design) matrix :math:`F`.
        L: Noise effect matrix :math:`L`.
        Qc: Spectral density :math:`Q_c`.
        dt: Time step :math:`\Delta t = X_2 - X_1`.

    Returns:
        Dictionary of named submatrices of the Van Loan exponential.
    """
    QL = L @ Qc @ L.T
    b = len(F)  # block size
    I = jnp.eye(b)
    Z = jnp.zeros_like(F)
    C = jnp.block(
        [
            [-F, I, Z, Z],
            [Z, -F, QL, Z],
            [Z, Z, F.T, I],
            [Z, Z, Z, Z],
        ]
    )
    VanLoanBlock = block_view(expm(C * dt), b)

    F1 = VanLoanBlock[0, 0]
    G1 = VanLoanBlock[0, 1]
    H1 = VanLoanBlock[0, 2]
    K1 = VanLoanBlock[0, 3]
    F2 = VanLoanBlock[1, 1]
    G2 = VanLoanBlock[1, 2]
    H2 = VanLoanBlock[1, 3]
    F3 = VanLoanBlock[2, 2]
    G3 = VanLoanBlock[2, 3]
    F4 = VanLoanBlock[3, 3]

    return {
        "F1": F1,
        "F2": F2,
        "F3": F3,
        "F4": F4,
        "G1": G1,
        "G2": G2,
        "G3": G3,
        "H1": H1,
        "H2": H2,
        "K1": K1,
    }
