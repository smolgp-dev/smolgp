import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import expm
from tinygp.helpers import JAXArray


def count_min_instids(t: JAXArray, delta: JAXArray) -> int:
    r"""Minimum number of ``instid`` groups needed for ``M`` exposure windows
    :math:`(t_i - \delta_i/2,\; t_i + \delta_i/2)` with arbitrary overlap.

    This is the chromatic number of an interval graph, which equals its maximum
    clique. See https://en.wikipedia.org/wiki/Interval_graph.
    
    Note for exposure windows,
    - :math:`b_i = a_j` is not an overlap and can use the same group id.
    - A zero-width window (``delta == 0``) strictly inside another window's
      span **must** conflict, since its readout would otherwise corrupt the
      enclosing exposure's running integra. However, two *coincident* 
      zero-width windows do not conflict with each other.

    So instead of a sweep, count for each window ``j`` the windows still open
    at :math:`a_j`, itself included, and take the largest.

    Cost is :math:`O(M \log M)`, dominated by the sorts.

    Deliberately numpy rather than ``jnp``: the result sizes the augmented
    state (see :func:`~smolgp.solvers.sample.merge_exposure_test_coords`), so
    it must be a concrete Python ``int`` and cannot be produced under ``jit``.

    Args:
        t: Exposure midpoints, length ``M``.
        delta: Exposure widths, length ``M`` (must be >= 0).

    Returns:
        The minimum number of groups, i.e. ``int(jnp.max(instid)) + 1`` for the
        assignment :func:`assign_instids` produces.
    """
    if isinstance(t, jax.core.Tracer) or isinstance(delta, jax.core.Tracer):
        raise TypeError(
            "count_min_instids needs concrete coordinates: it reads their values "
            "to size the augmented state, which cannot happen inside jit. Compute "
            "it outside the trace and pass it in (as GaussianProcess.sample's "
            "num_test_insts argument) -- it is 1 for instantaneous or "
            "non-overlapping test points."
        )
    a = np.asarray(t, dtype=float) - np.asarray(delta, dtype=float) / 2
    b = np.asarray(t, dtype=float) + np.asarray(delta, dtype=float) / 2
    if a.size == 0:
        return 0
    open_at_start = np.searchsorted(np.sort(a), a, side="right") - np.searchsorted(
        np.sort(b), a, side="right"
    )
    # A zero-width window has b_j == a_j, so the subtraction just counted it as
    # already closed; add it back so it still occupies a group of its own.
    return int(np.max(open_at_start + (b == a)))


def assign_instids(t: JAXArray, delta: JAXArray, num_insts: int) -> JAXArray:
    r"""Assign each exposure window to one of ``num_insts`` groups such that no
    two conflicting windows share a group.

    Unlike :func:`count_min_instids` this is fully jittable, provided
    ``num_insts`` is a static Python ``int``.

    Since the count is already known, simply sweep the windows in order of start 
    time and give each one *any* currently-free group. Optimal and vectorizes.

    Cost is :math:`O(M \log M)` for the sort plus :math:`O(M \cdot n)`, and beats
    the :math:`O(M \log M)` for an eager heap because the heap's cost is Python 
    interpreter overhead rather than its asymptotics, except only for very large 
    ``M``, and the heap is not jittable.

    Args:
        t: Exposure midpoints, length ``M``.
        delta: Exposure widths, length ``M`` (must be >= 0).
        num_insts: Number of groups to assign into; must be static, and at
            least :func:`count_min_instids` or the assignment is not valid.

    Returns:
        Length ``M`` integer array of group assignments,
        ``0 <= instid[i] < num_insts``.
    """
    a = t - delta / 2
    b = t + delta / 2
    order = jnp.argsort(a)
    a_sorted, b_sorted = a[order], b[order]

    def step(free_at, i):
        # free_at[g] is when group g's last window ended; -inf = never used.
        gid = jnp.argmax(free_at <= a_sorted[i])
        return free_at.at[gid].set(b_sorted[i]), gid

    _, gids = lax.scan(step, jnp.full(num_insts, -jnp.inf), jnp.arange(a.shape[0]))
    # gids are in start order; instid must come back in input order.
    return jnp.zeros_like(gids).at[order].set(gids)


def assign_min_instids(t: JAXArray, delta: JAXArray) -> tuple[JAXArray, int]:
    r"""
    Compute the minimum number of non-overlapping ``instid`` groups for a
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
    if it's  already finished" greedy sweep. Cost is :math:`O(M \log M)`.

    For calls inside ``jit``, first call :func:`count_min_instids` once outside the 
    trace and then call :func:`assign_instids` directly.

    Returns:
        instid: Length ``M`` integer array of group assignments.
        num_insts: The number of distinct groups used.
    """
    num_insts = count_min_instids(t, delta)
    return assign_instids(t, delta, num_insts), num_insts


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


def transition_sequence(A, Q, t: JAXArray) -> tuple[JAXArray, JAXArray]:
    r"""Per-step transition matrices and process noise, for precomputing.

    Returns ``A(0, Delta_k)`` and ``Q(0, Delta_k)`` for every step ``k``, with
    :math:`\Delta_0 = 0` for the first step (transition from the prior) and
    :math:`\Delta_k = t_k - t_{k-1}` thereafter.

    Building these with :func:`jax.vmap` ahead of the scan, and streaming them
    in as scan ``xs``, is much faster than calling ``A``/``Q`` inside the scan
    body even though the arithmetic is identical: both are matrix exponentials
    (or Van Loan blocks) of a fixed generator, so as a scan body they are a
    serial chain of small unfusable kernels, whereas vmapped they become one
    batched kernel. The tradeoff is carrying two extra arrays of shape 
    ``(N, dim, dim)`` in memory, which is not a significant addition.
    """
    Deltas = jnp.concatenate([jnp.zeros((1,), t.dtype), jnp.diff(t)])
    A_all = jax.vmap(lambda d: A(0, d))(Deltas)
    Q_all = jax.vmap(lambda d: Q(0, d))(Deltas)
    return A_all, Q_all


def kalman_gain(S: JAXArray, PHt: JAXArray) -> JAXArray:
    r"""The Kalman gain 
    
    .. math:: 
        K_k = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}.

    Args:
        S: The :math:`D \times D` innovation covariance  :math:`\mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k`
        PHt: The product :math:`\mathbf{P}_k^- \mathbf{H}_k^T`, shape ``(dim, D)``.

    For a scalar observation (``D == 1``) the inverse is just division,
    and ``D = S.shape[0]`` is a static shape known at trace time, so we
    can boost performance for this common case by avoiding ``solve``.
    """
    if S.shape[0] == 1:
        return PHt / S[0, 0]
    return jnp.linalg.solve(S.T, PHt.T).T


def smoothing_gain(P_pred_next: JAXArray, PAt: JAXArray) -> JAXArray:
    r"""Computes the RTS smoothing gain :math:`G_k` from

    .. math::

        G_k = \mathbf{P}_k \mathbf{A}_k^T \left[\mathbf{P}_{k+1}^{-}\right]^{-1}.

    Args:
        P_pred_next: The predicted covariance :math:`\mathbf{P}_{k+1}^{-}` for the next step.
        PAt: The product :math:`\mathbf{P}_k \mathbf{A}_k^T` for the current step.

    Returns:
        The RTS smoothing gain :math:`G_k`.

    TLDR; usage converts
        ``G_k = jnp.linalg.solve(P_pred_next.T, (P_k @ A_k.T).T).T``
    to
        ``G_k = get_smoothing_gain(P_pred_next, P_k @ A_k.T)``

    Guards against :math:`\mathbf{P}_{\mathrm{pred,next}}` being exactly singular. This
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
    """
    n = P_pred_next.shape[0]
    scale = jnp.trace(P_pred_next) / n
    sign, logdet = jnp.linalg.slogdet(P_pred_next)
    logdet_normalized = logdet - n * jnp.log(scale)
    is_singular = (sign <= 0) | (logdet_normalized < -30.0)

    def solve_generic(_):
        return jnp.linalg.solve(P_pred_next.T, PAt.T).T

    def solve_degenerate(_):
        Y, *_ = jnp.linalg.lstsq(P_pred_next.T, PAt.T)
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
