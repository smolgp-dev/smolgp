"""Generate posterior_sample_animation.gif for the sample.ipynb tutorial:
a step-by-step, narrated build-up of one state-space GP *posterior* sample,
extending make_sample_animation.py's prior-sampling storyboard with
Matheron's rule, split into its two real algorithmic passes:

    Pass A:          a plain forward SDE simulation (the prior draw), jointly
                      over the sample coordinates AND every data time
                      (including one deliberately NOT coincident with any
                      sample coordinate, to show it still enters the scan).
    Pass B, part 1:  condition() -- a full Kalman-filter-forward +
                      RTS-smoother-backward pass, computed ONLY at the data
                      times, one at a time in this visualization. Sample
                      states with no data of their own are untouched here.
    Pass B, part 2:  predict() -- propagates that correction out to every
                      other (non-data) sample state, via the same
                      retrodict/interpolate/extrapolate machinery used
                      everywhere else in the tutorial.

Not run as part of the docs build -- run manually whenever the animation
needs regenerating:

    python docs/tutorials/sample_anim/make_posterior_sample_animation.py

and commit the resulting posterior_sample_animation.gif alongside the notebook.
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from anim_utils import ANIM_DPI, FPS, HERE, Timing, write_interactive_html
from matplotlib.animation import FuncAnimation

import smolgp
from smolgp.helpers import robust_sqrt
from smolgp.solvers.sample import sample_prior_trajectory
from smolgp.solvers.state_coords import StateCoords

jax.config.update("jax_enable_x64", True)
mpl.rc("font", family="sans serif", size=16)

# ---------------------------------------------------------------------------
# PACING -- how long each key moment lasts, in seconds. This is the one block
# to edit when re-timing the animation; the storyboard below refers to these
# by name. Repeated hold frames are deduplicated when the player is written,
# so lengthening an explanatory beat costs runtime but almost no file size.
#
# Pass A repeats once per timeline segment and Pass B once per data point, so
# each has a separate "first" timing (where the mechanic is taught) and a much
# brisker one for the repeats that follow.
# ---------------------------------------------------------------------------
TIMING = Timing(
    blank=0.4,  # empty axes before anything is drawn
    true_process=0.6,  # the (unknowable) curve the data came from
    data_reveal=1.0,  # noisy observations appear on it
    coords=1.5,  # the sample coordinates alone (solid lines)
    coords_merged=1.0,  # ...then the data-only time joins them (dashed)
    truth_fade=0.4,  # the true curve fades away...
    # Pass A: the prior draw. Already shown in full in the prior-sampling
    # animation, so this is a brisk recap under a single title.
    passA_scatter=0.6,
    passA_scatter_cloud=0.3,
    passA_scatter_chosen=0.5,
    passA_walk=0.15,
    passA_walk_cloud=0.05,
    passA_walk_chosen=0.1,
    prior_done=2.0,  # the finished prior draw, not yet matching the data
    # Pass B part 1: condition, one data point at a time.
    first_highlight=2.5,  # bold the data point being compared
    first_noise=1.5,  # show the observation-noise distribution (red bar)
    first_perturb=1.0,  # ...and move the sampled point by one draw from it
    first_residual=1.5,  # draw the residual between draw+noise and the data
    first_snap=1.0,  # filter + smooth pulls that marker into place
    first_settled=1.0,  # corrected at this data point only
    highlight=0.3,  # ...and the same four beats for the remaining points
    noise=0.35,
    perturb=0.3,
    residual=0.5,
    snap=0.5,
    settled=0.35,
    # Pass B part 2: predict, propagating the correction everywhere else.
    predict_pull=1.5,
    predict_done=0.5,
    posterior_done=0.25,  # the finished posterior sample
    fade_out=0.25,
    final=2.0,
    # Silent pause held on the last frame of a key moment, so the viewer has
    # time to digest what just happened before the next thing starts moving.
    beat=1.0,
    beat_short=0.2,  # between the brisk, repeated segments/data points
)

# Same damped oscillator as the rest of the tutorial (and the prior-sampling animation)
sigma, omega, quality = 2.1, 2 * jnp.pi / 60.0, 5.3
kernel = smolgp.kernels.SHO(omega, quality, sigma)
PERIOD = 2 * jnp.pi / omega  # 60 sec

T_MAX = float(2 * PERIOD)  # span two timescales
N_COORDS = 16  # number of "sample coordinates"
N_SCATTER = 50  # candidate draws shown at the very first point
N_PATHS = 10  # candidate wandering paths shown per segment
N_SUB = 40  # points per dense sub-grid segment (path smoothness)
TRUTH_GRID_N = 250  # fine grid for the illustrative "true process" curve
SEED = 21  # different draw than the prior-sampling animation

# Data at 4 of the 16 sample coordinates, PLUS one deliberately NOT
# coincident with any sample coordinate (between t_coords[5]=40 and
# t_coords[6]=48) -- this is what lets Pass B part 1 (condition, touches
# only data times) and part 2 (predict, reaches every other sample state)
# be shown as genuinely different operations.
DATA_COINCIDE_IDX = [2, 8, 11, 13]
DATA_EXTRA_T = [44.0]

t_coords = jnp.linspace(0.0, T_MAX, N_COORDS)
t_coords_np = np.asarray(t_coords)
key = jax.random.PRNGKey(SEED)

# ---------------------------------------------------------------------------
# 0. The merged "boundary" timeline: every sample coordinate, plus the one
# extra data-only time, sorted together. Built by explicit bookkeeping (not
# float comparison) so each entry's provenance -- "sample" (one of the 16
# requested coordinates) vs "data_only" (enters the simulation but is never
# an output sample) -- is known exactly.
# ---------------------------------------------------------------------------
boundary_entries = [(float(t), "sample", j) for j, t in enumerate(t_coords_np)]
boundary_entries += [(float(t), "data_only", None) for t in DATA_EXTRA_T]
boundary_entries.sort(key=lambda e: e[0])
N_BOUND = len(boundary_entries)
boundary_t_np = np.array([e[0] for e in boundary_entries])
boundary_kind = [e[1] for e in boundary_entries]

data_boundary_pos = []
for idx in DATA_COINCIDE_IDX:
    pos = next(
        i for i, e in enumerate(boundary_entries) if e[1] == "sample" and e[2] == idx
    )
    data_boundary_pos.append(pos)
for t_extra in DATA_EXTRA_T:
    pos = next(
        i
        for i, e in enumerate(boundary_entries)
        if e[1] == "data_only" and e[0] == t_extra
    )
    data_boundary_pos.append(pos)
data_boundary_pos = np.array(data_boundary_pos)
# Process data points in chronological order (matching the forward-in-time
# nature of the Kalman filter sweep) -- NOT construction order, which
# interleaves the coincident and non-coincident times arbitrarily.
data_boundary_pos = data_boundary_pos[np.argsort(boundary_t_np[data_boundary_pos])]
N_DATA = len(data_boundary_pos)
t_data_np = boundary_t_np[data_boundary_pos]
t_data = jnp.array(t_data_np)

has_data = np.zeros(N_BOUND, dtype=bool)
data_index_of_boundary = np.full(N_BOUND, -1, dtype=int)
for m, pos in enumerate(data_boundary_pos):
    has_data[pos] = True
    data_index_of_boundary[pos] = m

var0 = float(kernel(t_coords[:1], t_coords[:1])[0, 0])
NOISE_STD = 0.3 * float(np.sqrt(var0))

# ---------------------------------------------------------------------------
# 0b. An independent "true process" draw, for illustration only -- shown as
# a faded grey curve, then used to generate the (fixed) noisy data. The
# algorithm itself never sees this curve; only the data and the desired
# sample coordinates.
# ---------------------------------------------------------------------------
truth_entries = [
    (float(t), "grid", None) for t in np.linspace(0.0, T_MAX, TRUTH_GRID_N)
]
truth_entries += [(float(t), "data", m) for m, t in enumerate(t_data_np)]
truth_entries.sort(key=lambda e: e[0])
t_truth_dense_np = np.array([e[0] for e in truth_entries])
truth_data_pos = [None] * N_DATA
for i, e in enumerate(truth_entries):
    if e[1] == "data":
        truth_data_pos[e[2]] = i

t_truth_dense = jnp.array(t_truth_dense_np)
K_truth = t_truth_dense.shape[0]
state_coords_truth = StateCoords.instantaneous(t_truth_dense)
key, sub = jax.random.split(key)
x_traj_truth = sample_prior_trajectory(kernel, state_coords_truth, sub)
H_truth = jax.vmap(kernel.observation_model)(t_truth_dense)
y_truth_dense_np = np.asarray(jnp.einsum("kij,kj->ki", H_truth, x_traj_truth)[:, 0])
y_truth_at_data = y_truth_dense_np[np.array(truth_data_pos)]

key, sub = jax.random.split(key)
y_data = y_truth_at_data + NOISE_STD * np.asarray(jax.random.normal(sub, (N_DATA,)))

# ---------------------------------------------------------------------------
# 1. Draw ONE true, fully continuous joint sample of the LATENT STATE across
# the whole domain (Pass A), via sample_prior_trajectory directly -- the
# merged boundary timeline from Section 0 means the data-only time is
# simulated through exactly like any other point, even though it will never
# be an output sample.
# ---------------------------------------------------------------------------
grid_pieces = []
boundary_idx = [0]
offset = 0
for i in range(N_BOUND - 1):
    t0, t1 = boundary_t_np[i], boundary_t_np[i + 1]
    piece = jnp.linspace(t0, t1, N_SUB)
    if i > 0:
        piece = piece[1:]  # don't duplicate the shared boundary point
    grid_pieces.append(piece)
    offset += piece.shape[0]
    boundary_idx.append(offset - 1)
t_dense_full = jnp.concatenate(grid_pieces)
t_dense_full_np = np.asarray(t_dense_full)
K = t_dense_full.shape[0]
boundary_idx_arr = np.array(boundary_idx)
data_dense_idx = boundary_idx_arr[
    data_boundary_pos
]  # index into t_dense_full for each data point

state_coords = StateCoords.instantaneous(t_dense_full)
key, sub = jax.random.split(key)
x_traj_true = sample_prior_trajectory(kernel, state_coords, sub)  # (K, dim)

H_all = jax.vmap(kernel.observation_model)(t_dense_full)  # (K, 1, dim)
y_true_dense = jnp.einsum("kij,kj->ki", H_all, x_traj_true)[:, 0]
y_true_dense_np = np.asarray(y_true_dense)

y_all_bound_np = y_true_dense_np[
    boundary_idx_arr
]  # raw draw's value at each of the N_BOUND boundary points


# ---------------------------------------------------------------------------
# 2. Per segment: honest alternate "what if" continuations, generated via the
# kernel's own transition_matrix/process_noise recursion -- identical to
# make_sample_animation.py's Section 2, just over N_BOUND-1 segments.
# ---------------------------------------------------------------------------
def simulate_from(x_start, sub_t, key):
    A, Q = kernel.transition_matrix, kernel.process_noise

    def step(x_prev, inputs):
        k, key_k = inputs
        dt = jnp.where(k > 0, sub_t[k] - sub_t[jnp.maximum(k - 1, 0)], 0.0)
        A_k, Q_k = A(0, dt), Q(0, dt)
        z = jax.random.normal(key_k, shape=x_start.shape)
        x_k = jnp.where(k > 0, A_k @ x_prev + robust_sqrt(Q_k) @ z, x_start)
        return x_k, x_k

    keys = jax.random.split(key, sub_t.shape[0])
    _, x_path = jax.lax.scan(step, x_start, (jnp.arange(sub_t.shape[0]), keys))
    return jax.vmap(lambda t, x: kernel.observation_model(t) @ x)(sub_t, x_path)[:, 0]


sub_grids = []
candidate_paths = []  # list of (n_seg, N_PATHS) arrays
chosen_path_idx = []  # which column holds the TRUE continuation, per segment

for i in range(N_BOUND - 1):
    a, b = boundary_idx[i], boundary_idx[i + 1]
    sub_t = t_dense_full[a : b + 1]
    x_start = x_traj_true[a]
    true_y = y_true_dense[a : b + 1]

    key, sub = jax.random.split(key)
    idx = int(jax.random.randint(sub, (), 0, N_PATHS))
    paths = np.zeros((sub_t.shape[0], N_PATHS))
    for k in range(N_PATHS):
        if k == idx:
            paths[:, k] = np.asarray(true_y)
        else:
            key, sub = jax.random.split(key)
            paths[:, k] = np.asarray(simulate_from(x_start, sub_t, sub))

    sub_grids.append(np.asarray(sub_t))
    candidate_paths.append(paths)
    chosen_path_idx.append(idx)

# scatter0: candidate initial-state draws, true initial position mixed in at a random slot
key, sub = jax.random.split(key)
scatter0 = np.array(jax.random.normal(sub, (N_SCATTER,)) * np.sqrt(var0))
key, sub = jax.random.split(key)
chosen0_idx = int(jax.random.randint(sub, (), 0, N_SCATTER))
scatter0[chosen0_idx] = y_all_bound_np[0]

# ---------------------------------------------------------------------------
# 3. Matheron's rule, split into its real two passes.
#
# Pass B part 1 (condition): forms a residual at each of the N_DATA data
# times (real data minus the raw draw projected there, plus a fresh
# observation-noise draw) and runs it through the Kalman filter + RTS
# smoother -- computed ONCE, up front, via growing prefixes of the same
# public condition()/predict() API used elsewhere (each prefix's result is a
# genuine posterior mean given that data subset; the final one, using all
# N_DATA points, is mathematically identical to what
# GaussianProcess.sample()'s internal condition_batched_mean would produce,
# verified separately to match to machine precision).
#
# Pass B part 2 (predict): the SAME final correction function, evaluated at
# every OTHER (non-data) point on the dense grid -- this is what actually
# reaches sample states that have no data of their own.
# ---------------------------------------------------------------------------
prior_obs_at_data = y_all_bound_np[data_boundary_pos]
key, sub = jax.random.split(key)
noise_sample = np.asarray(NOISE_STD * jax.random.normal(sub, (N_DATA,)))
noisy_obs_at_data = prior_obs_at_data + noise_sample
residual = y_data - noisy_obs_at_data

correction_curves = [np.zeros_like(t_dense_full_np)]  # k=0: no data incorporated yet
for k in range(1, N_DATA + 1):
    if k == 1:
        # ConditionedStates.project_at_data's .squeeze() collapses a length-1
        # result to a 0-d scalar for N=1 (a pre-existing library edge case);
        # duplicating the single anchor point sidesteps it without changing
        # the conditioning result (two identical noisy observations of the
        # same value is a standard, well-defined Bayesian update).
        t_k = jnp.array([t_data_np[0], t_data_np[0]])
        y_k = jnp.array([residual[0], residual[0]])
        noise_k = jnp.full(2, NOISE_STD**2)
    else:
        t_k = t_data[:k]
        y_k = jnp.asarray(residual[:k])
        noise_k = jnp.full(k, NOISE_STD**2)
    gp_prefix = smolgp.GaussianProcess(kernel, X=t_k, noise=noise_k)
    _, cond_prefix = gp_prefix.condition(y_k)
    mean_prefix_dense = np.asarray(cond_prefix.predict(t_dense_full))
    correction_curves.append(mean_prefix_dense)

FULL_CORR = correction_curves[
    -1
]  # the true, all-data-conditioned correction, on the dense grid

# ---------------------------------------------------------------------------
# Fixed y-axis range for the whole animation
# ---------------------------------------------------------------------------
all_vals = np.concatenate(
    [scatter0]
    + [p.ravel() for p in candidate_paths]
    + [y_data - NOISE_STD, y_data + NOISE_STD]
    + [y_truth_dense_np]
    + [y_true_dense_np + c for c in correction_curves]
)
ylim = (float(all_vals.min()) - 0.5, float(all_vals.max()) + 0.5)


def ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def render_state(dense_frac, data_corr_by_k):
    """dense_frac in [0,1]: how much of FULL_CORR to show on the continuous
    curve and on sample states with no data of their own (0 before Pass B
    part 2 starts). data_corr_by_k: length-N_DATA array of each data
    point's OWN current correction (frozen once its own turn settles)."""
    dense_corr = dense_frac * FULL_CORR
    bound_corr = dense_corr[boundary_idx_arr].copy()
    for m in range(N_DATA):
        bound_corr[data_boundary_pos[m]] = data_corr_by_k[m]
    return dense_corr, bound_corr


ZERO_DATA_CORR = np.zeros(N_DATA)

# ---------------------------------------------------------------------------
# Build the scene list
# ---------------------------------------------------------------------------
scenes = []


def add_hold(spec, moment):
    """Hold `spec` on screen for TIMING[moment] seconds."""
    for _ in range(TIMING.frames(moment)):
        scenes.append(dict(spec))


def sweep(moment):
    """Fractions 0..1 spanning an animated transition of TIMING[moment] sec."""
    return np.linspace(0.0, 1.0, TIMING.frames(moment))


def add_beat(moment="beat"):
    """Hang on whatever is currently on screen, letting the viewer digest the
    moment that just finished before the next one starts moving."""
    last = dict(scenes[-1])
    for _ in range(TIMING.frames(moment)):
        scenes.append(dict(last))


add_hold({"stage": "blank", "label": ""}, "blank")
add_hold(
    {
        "stage": "true_process",
        "label": "The true process (for illustration only)",
    },
    "true_process",
)
add_beat()
add_hold({"stage": "data_reveal", "label": "Observed, noisy data"}, "data_reveal")
add_beat()
add_hold(
    {
        "stage": "coords",
        "label": "Sample coordinates (can be anywhere)",
    },
    "coords",
)
add_beat()
add_hold(
    {
        "stage": "coords_merged",
        "label": "Merge sample and data coordinates",
    },
    "coords_merged",
)
add_beat()
# The true curve fades away under the preceding title, which already says
# what is left behind -- no separate held frame needed for that.
for frac in 1.0 - sweep("truth_fade"):
    scenes.append(
        {
            "stage": "fade_truth",
            "truth_alpha": frac,
            "label": "Merge sample and data coordinates",
        }
    )
add_beat()

## Pass A: quick run-through (steps 1-3 already shown in full in the prior-
## sampling animation), all under one title.
PASS_A_LABEL = "Pass A: Draw a prior sample"
for frac in 0.3 + 0.7 * sweep("passA_scatter"):
    n_shown = max(1, int(frac * N_SCATTER))
    scenes.append({"stage": "scatter0", "label": PASS_A_LABEL, "n_shown": n_shown})
add_hold(
    {"stage": "scatter0", "label": PASS_A_LABEL, "n_shown": N_SCATTER},
    "passA_scatter_cloud",
)
add_hold({"stage": "scatter0_chosen", "label": PASS_A_LABEL}, "passA_scatter_chosen")

for i in range(N_BOUND - 1):
    n_pts_total = sub_grids[i].shape[0]
    for frac in 0.3 + 0.7 * sweep("passA_walk"):
        n_pts = max(2, int(frac * n_pts_total))
        scenes.append(
            {"stage": "walk", "seg": i, "label": PASS_A_LABEL, "n_pts": n_pts}
        )
    add_hold(
        {"stage": "walk", "seg": i, "label": PASS_A_LABEL, "n_pts": n_pts_total},
        "passA_walk_cloud",
    )
    add_hold(
        {"stage": "walk_chosen", "seg": i, "label": PASS_A_LABEL}, "passA_walk_chosen"
    )

add_hold(
    {
        "stage": "prior_done",
        "label": "Prior sample obtained, now condition on the data",
    },
    "prior_done",
)
add_beat()

## Pass B, part 1: condition -- one data point at a time. Only that data
## point's own marker moves; everything else (the curve, and every sample
## state without data) stays exactly at its raw value.
_data_corr_state = ZERO_DATA_CORR.copy()

for k in range(1, N_DATA + 1):
    m = k - 1  # index into t_data / y_data / data_boundary_pos
    is_first = m == 0
    after_val = correction_curves[k][data_dense_idx[m]]

    def snapshot(overrides=None):
        d = _data_corr_state.copy()
        if overrides is not None:
            d[m] = overrides
        return d

    # The residual is formed from the RAW prior draw (that is what
    # sample() actually differences against the data), so this point starts
    # at its raw value, is displaced by exactly its own noise draw, and only
    # then gets pulled to the conditioned position.
    noise_val = float(noise_sample[m])

    add_hold(
        {
            "stage": "highlight_data",
            "k": m,
            "data_corr": snapshot(0.0),
            "label": "Pass B, part 1: condition (at observed data)",
        },
        "first_highlight" if is_first else "highlight",
    )
    add_hold(
        {
            "stage": "noise_errorbar",
            "k": m,
            "data_corr": snapshot(0.0),
            "label": "Add random observation noise",
        },
        "first_noise" if is_first else "noise",
    )
    for frac in sweep("first_perturb" if is_first else "perturb"):
        w = ease(frac)
        scenes.append(
            {
                "stage": "perturb",
                "k": m,
                "data_corr": snapshot(w * noise_val),
                "label": "Add random observation noise",
            }
        )
    add_hold(
        {
            "stage": "residual",
            "k": m,
            "data_corr": snapshot(noise_val),
            "label": "Form the residual",
        },
        "first_residual" if is_first else "residual",
    )
    for frac in sweep("first_snap" if is_first else "snap"):
        w = ease(frac)
        scenes.append(
            {
                "stage": "snap",
                "k": m,
                "data_corr": snapshot((1 - w) * noise_val + w * after_val),
                "label": "Filter + smooth",
            }
        )
    _data_corr_state[m] = after_val
    add_hold(
        {
            "stage": "settled",
            "k": m,
            "data_corr": snapshot(),
            "label": "Data point conditioned",
        },
        "first_settled" if is_first else "settled",
    )
    # Long pause after the first data point (the teaching pass) and after the
    # last one (before the correction propagates); a breath between the rest.
    add_beat("beat" if is_first or m == N_DATA - 1 else "beat_short")

## Pass B, part 2: predict -- propagate the correction to every other
## sample state (and the continuous curve), shown as one single "pull".
for frac in sweep("predict_pull"):
    w = ease(frac)
    scenes.append(
        {
            "stage": "predict_pull",
            "dense_frac": w,
            "data_corr": _data_corr_state,
            "label": "Pass B, part 2: predict (propagate to rest)",
        }
    )
add_hold(
    {
        "stage": "predict_pull",
        "dense_frac": 1.0,
        "data_corr": _data_corr_state,
        "label": "Pass B, part 2: predict (propagate to rest)",
    },
    "predict_done",
)
add_beat()

add_hold(
    {
        "stage": "posterior_done",
        "dense_frac": 1.0,
        "data_corr": _data_corr_state,
        "label": "One posterior sample",
    },
    "posterior_done",
)
add_beat()

for alpha in 1.0 - sweep("fade_out"):
    scenes.append(
        {
            "stage": "fade",
            "dense_frac": 1.0,
            "data_corr": _data_corr_state,
            "alpha": alpha,
            "label": "Final posterior sample",
        }
    )
add_hold(
    {
        "stage": "final",
        "dense_frac": 1.0,
        "data_corr": _data_corr_state,
        "label": "Final posterior sample",
    },
    "final",
)

N_FRAMES = len(scenes)

# ---------------------------------------------------------------------------
# 4. Render
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.5), dpi=ANIM_DPI)
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.22, top=0.85)


def draw_coord_lines(bold_idx=None, include_data_only=True, alpha_scale=1.0):
    """Vertical line per state in the merged timeline: solid for a requested
    sample coordinate, dashed for a data time that is not one of them.

    ``include_data_only=False`` draws only the sample coordinates, for the
    opening beat that introduces them before the data times are merged in.
    ``alpha_scale`` fades the whole scaffold out at the end, once the states
    it was there to mark have been drawn.
    """
    if alpha_scale <= 0.0:
        return
    for j in range(N_BOUND):
        tj = boundary_t_np[j]
        is_sample = boundary_kind[j] == "sample"
        if bold_idx is not None and j == bold_idx:
            ax.axvline(tj, color="0.2", lw=2, alpha=0.85 * alpha_scale, zorder=1)
        elif is_sample:
            ax.axvline(tj, color="0.5", lw=1.3, alpha=0.45 * alpha_scale, zorder=1)
        elif include_data_only:
            ax.axvline(
                tj, color="0.5", lw=1.3, ls="--", alpha=0.45 * alpha_scale, zorder=1
            )


def draw_true_process(alpha):
    if alpha > 0:
        ax.plot(
            t_truth_dense_np,
            y_truth_dense_np,
            "-",
            color="0.65",
            lw=1.5,
            alpha=alpha,
            zorder=2,
        )


def draw_data_points(highlight=None, alpha=1.0):
    """The observed data. ``alpha`` fades them out at the very end, leaving
    only the sampled states -- the sample is the deliverable; the data were
    the input."""
    if alpha <= 0.0:
        return
    ax.errorbar(
        t_data_np,
        y_data,
        yerr=NOISE_STD,
        fmt="D",
        color="black",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        ms=8,
        alpha=alpha,
        zorder=11,
    )
    if highlight is not None:
        ax.plot(
            [t_data_np[highlight]],
            [y_data[highlight]],
            "D",
            color="black",
            ms=13,
            zorder=12,
            mec="gold",
            mew=2,
        )


def draw_resolved_paths(upto_seg, alpha=0.6):
    for seg in range(upto_seg):
        idx = chosen_path_idx[seg]
        ax.plot(
            sub_grids[seg],
            candidate_paths[seg][:, idx],
            color="crimson",
            alpha=alpha,
            lw=2,
            zorder=6,
        )


def draw_markers(bound_corr, upto=N_BOUND - 1, hollow_alpha=1.0, solid_alpha=1.0):
    for j in range(upto + 1):
        t_j = boundary_t_np[j]
        y_j = y_all_bound_np[j] + bound_corr[j]
        if boundary_kind[j] == "sample":
            ax.plot(
                [t_j],
                [y_j],
                "o",
                color="crimson",
                alpha=solid_alpha,
                ms=9,
                zorder=10,
                mec="k",
                mew=1,
            )
        else:
            ax.plot(
                [t_j],
                [y_j],
                "o",
                mfc="none",
                mec="crimson",
                alpha=hollow_alpha,
                ms=9,
                mew=2,
                zorder=10,
            )


def draw_straight_chain(bound_corr, alpha=0.6):
    """Straight connectors between the sampled states.

    The smooth curve drawn until now is the underlying continuous process; the
    *sample* is just the states at the requested coordinates, so the ending
    swaps the curve for plain segments joining them. The data-only state is
    skipped -- it was simulated through, but is never an output sample.
    """
    if alpha <= 0.0:
        return
    sel = [j for j in range(N_BOUND) if boundary_kind[j] == "sample"]
    xs = boundary_t_np[sel]
    ys = np.array([y_all_bound_np[j] + bound_corr[j] for j in sel])
    ax.plot(xs, ys, "-", color="crimson", lw=1.5, alpha=alpha, zorder=9)


def draw_curve(dense_corr, alpha=1.0, lw=2):
    ax.plot(
        t_dense_full_np,
        y_true_dense_np + dense_corr,
        "-",
        color="crimson",
        alpha=alpha,
        lw=lw,
        zorder=8,
    )


def draw_noise_errorbar(k, muted=False):
    """The noise distribution N(0, R) that gets added to the RAW
    (uncorrected) prior draw at data point k, centered on that raw value --
    fixed, computed once up front, same regardless of which prefix has been
    conditioned on so far.

    ``muted`` recedes it into the background for the residual beat, where the
    (also orange, also vertical, same x) residual bar is the subject and would
    otherwise be indistinguishable from this."""
    t_k = t_data_np[k]
    center = prior_obs_at_data[k]
    # Red, matching the sampled point it belongs to (the orange is reserved
    # for the residual), and with no marker of its own -- draw_markers
    # already draws the point at this centre.
    ax.errorbar(
        [t_k],
        [center],
        yerr=[[NOISE_STD], [NOISE_STD]],
        fmt="none",
        ecolor="crimson",
        elinewidth=2.5,
        capsize=8,
        alpha=0.35 if muted else 1.0,
        zorder=12 if muted else 13,
    )


def update(frame_idx):
    ax.clear()
    ax.set_xlim(-2, T_MAX + 2)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("f(t)")
    spec = scenes[frame_idx]
    stage = spec["stage"]
    ax.set_title(spec["label"], fontsize=15)

    if stage == "blank":
        pass
    elif stage == "true_process":
        draw_true_process(1.0)
    elif stage == "data_reveal":
        draw_true_process(1.0)
        draw_data_points()
    elif stage == "coords":
        # Only the requested sample coordinates yet -- the data-only time
        # joins the timeline in the next beat.
        draw_true_process(1.0)
        draw_data_points()
        draw_coord_lines(include_data_only=False)
    elif stage == "coords_merged":
        draw_true_process(1.0)
        draw_data_points()
        draw_coord_lines()
    elif stage == "fade_truth":
        draw_true_process(spec["truth_alpha"])
        draw_data_points()
        draw_coord_lines()
    elif stage == "scatter0":
        draw_coord_lines(bold_idx=0)
        draw_data_points()
        n = spec["n_shown"]
        ax.scatter(
            np.full(n, boundary_t_np[0]),
            scatter0[:n],
            color="C0",
            alpha=0.25,
            s=18,
            zorder=5,
        )
    elif stage == "scatter0_chosen":
        draw_coord_lines(bold_idx=0)
        draw_data_points()
        ax.scatter(
            np.full(N_SCATTER, boundary_t_np[0]),
            scatter0,
            color="C0",
            alpha=0.12,
            s=18,
            zorder=5,
        )
        draw_markers(np.zeros(N_BOUND), upto=0)
    elif stage == "walk":
        seg = spec["seg"]
        draw_coord_lines(bold_idx=seg + 1)
        draw_data_points()
        draw_resolved_paths(seg)
        draw_markers(np.zeros(N_BOUND), upto=seg)
        n_pts = spec["n_pts"]
        sg = sub_grids[seg][:n_pts]
        for k in range(N_PATHS):
            ax.plot(sg, candidate_paths[seg][:n_pts, k], color="C0", alpha=0.25, lw=1)
    elif stage == "walk_chosen":
        seg = spec["seg"]
        draw_coord_lines(bold_idx=seg + 1)
        draw_data_points()
        draw_resolved_paths(seg)
        idx = chosen_path_idx[seg]
        sg = sub_grids[seg]
        for k in range(N_PATHS):
            if k != idx:
                ax.plot(sg, candidate_paths[seg][:, k], color="C0", alpha=0.08, lw=1)
        ax.plot(
            sg,
            candidate_paths[seg][:, idx],
            color="crimson",
            alpha=0.9,
            lw=2.5,
            zorder=8,
        )
        draw_markers(np.zeros(N_BOUND), upto=seg + 1)
    elif stage == "prior_done":
        draw_coord_lines()
        draw_data_points()
        draw_curve(np.zeros(K))
        draw_markers(np.zeros(N_BOUND))
    elif stage in (
        "highlight_data",
        "noise_errorbar",
        "perturb",
        "residual",
        "snap",
        "settled",
    ):
        k = spec["k"]
        _, bound_corr = render_state(0.0, spec["data_corr"])
        draw_coord_lines(bold_idx=data_boundary_pos[k])
        draw_data_points(highlight=k)
        draw_curve(np.zeros(K))
        draw_markers(bound_corr)
        if stage in ("noise_errorbar", "perturb"):
            draw_noise_errorbar(k)
        elif stage == "residual":
            draw_noise_errorbar(k, muted=True)
            # The residual itself: the gap between the real datum and this
            # draw's synthetic noisy observation, drawn as the span between
            # them rather than as a marker at one end.
            t_k = t_data_np[k]
            y_noisy = noisy_obs_at_data[k]
            ax.plot(
                [t_k, t_k],
                [y_data[k], y_noisy],
                "-",
                color="darkorange",
                lw=4,
                solid_capstyle="butt",
                zorder=13,
            )
    elif stage == "predict_pull":
        dense_corr, bound_corr = render_state(spec["dense_frac"], spec["data_corr"])
        draw_coord_lines()
        draw_data_points()
        draw_curve(dense_corr)
        draw_markers(bound_corr)
    elif stage == "posterior_done":
        dense_corr, bound_corr = render_state(spec["dense_frac"], spec["data_corr"])
        draw_coord_lines()
        draw_data_points()
        draw_curve(dense_corr)
        draw_markers(bound_corr)
    elif stage == "fade":
        # The scaffolding (coordinate lines, continuous curve, and the
        # data-only state that was never an output) fades out as the straight
        # connectors between the sampled states fade in.
        alpha = spec.get("alpha", 0.0)
        _dense_corr, bound_corr = render_state(spec["dense_frac"], spec["data_corr"])
        draw_coord_lines(alpha_scale=alpha)
        draw_data_points(alpha=alpha)
        draw_curve(_dense_corr, alpha=alpha)
        draw_straight_chain(bound_corr, alpha=(1.0 - alpha) * 0.6)
        draw_markers(bound_corr, hollow_alpha=alpha, solid_alpha=1.0)
    elif stage == "final":
        _dense_corr, bound_corr = render_state(spec["dense_frac"], spec["data_corr"])
        draw_straight_chain(bound_corr)
        draw_markers(bound_corr, hollow_alpha=0.0, solid_alpha=1.0)

    return []


ani = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
write_interactive_html(ani, "posterior_sample_animation.html", fps=12)
ani.save(HERE / "posterior_sample_animation.gif", writer="pillow", fps=12)
print(f"Saved posterior_sample_animation.html and .gif ({N_FRAMES} frames)")
