"""Generate sample_animation.gif for the sample.ipynb tutorial's opening
cartoon: a step-by-step, narrated build-up of one state-space GP sample
path, illustrating sampling as literally simulating the forward SDE one
state at a time.

Not run as part of the docs build -- run manually whenever the animation
needs regenerating:

    python docs/tutorials/sample_anim/make_sample_animation.py

and commit the resulting sample_animation.gif alongside the notebook.
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
# to edit when re-timing the animation; everything below refers to these by
# name. Repeated hold frames are deduplicated when the player is written, so
# lengthening an explanatory beat costs runtime but almost no file size.
#
# The walk from one state to the next repeats N_COORDS-1 times, so its first
# occurrence (where the mechanic is actually being taught) is timed separately
# from the repeats that follow.
# ---------------------------------------------------------------------------
TIMING = Timing(
    blank=0.5,  # empty axes before anything is drawn
    coords=1.0,  # "Sample coordinates" -- the vertical lines appear
    draw_initial_state=1.0,  # candidate initial states fan in
    initial_state_cloud=1.0,  # full cloud of candidates, before one is picked
    initial_state_chosen=1.0,  # the chosen initial state, highlighted
    first_walk=1.0,  # first segment: candidate continuations sweep out
    first_walk_candidates=1.0,  # ...all candidates shown
    first_walk_chosen=1.0,  # ...one chosen, its endpoint marked
    walk=0.5,  # each later segment, same three beats but brisker
    walk_candidates=0.2,
    walk_chosen=0.5,
    fade_out=1.0,  # connecting paths fade, leaving the sampled states
    final=3.0,  # the finished realization
    # Silent pause held on the last frame of a key moment, so the viewer has
    # time to digest what just happened before the next thing starts moving.
    beat=1.0,
    beat_short=0.4,  # between the brisk, repeated walk segments
)

# Same damped oscillator as the rest of the tutorial
sigma, omega, quality = 2.1, 2 * jnp.pi / 60.0, 5.3
kernel = smolgp.kernels.SHO(omega, quality, sigma)
PERIOD = 2 * jnp.pi / omega  # 60 sec

T_MAX = float(2 * PERIOD)  # span two timescales -- comfortably more than one full cycle
N_COORDS = 12  # number of "sample coordinates"
N_SCATTER = 50  # candidate draws shown at the very first point
N_PATHS = 10  # candidate wandering paths shown per segment
N_SUB = 40  # points per dense sub-grid segment (path smoothness)
SEED = 7

t_coords = jnp.linspace(0.0, T_MAX, N_COORDS)
key = jax.random.PRNGKey(SEED)

# ---------------------------------------------------------------------------
# 1. Draw ONE true, fully continuous joint sample of the LATENT STATE
# (position AND velocity) across the whole domain, via sample_prior_trajectory
# directly -- the same function GaussianProcess.sample() calls internally,
# just exposing the full state instead of only the observed projection.
#
# This is NOT equivalent to independently re-conditioning each segment on
# only the observed position at the previous chosen point (what an earlier
# version of this script did): the SHO kernel's state is 2D (position AND
# velocity), so conditioning on position alone leaves velocity undetermined
# -- continuing "fresh" from just an observed value gives a real derivative
# discontinuity at every join. Slicing up one genuine joint sample instead
# is differentiable everywhere by construction.
# ---------------------------------------------------------------------------
grid_pieces = []
boundary_idx = [0]
offset = 0
for i in range(N_COORDS - 1):
    t0, t1 = t_coords[i], t_coords[i + 1]
    piece = jnp.linspace(t0, t1, N_SUB)
    if i > 0:
        piece = piece[1:]  # don't duplicate the shared boundary point
    grid_pieces.append(piece)
    offset += piece.shape[0]
    boundary_idx.append(offset - 1)
t_dense_full = jnp.concatenate(grid_pieces)
K = t_dense_full.shape[0]

state_coords = StateCoords.instantaneous(t_dense_full)
key, sub = jax.random.split(key)
x_traj_true = sample_prior_trajectory(
    kernel, state_coords, sub
)  # (K, dim): position, velocity

H_all = jax.vmap(kernel.observation_model)(t_dense_full)  # (K, 1, dim)
y_true_dense = jnp.einsum("kij,kj->ki", H_all, x_traj_true)[:, 0]

y_chosen = np.asarray(y_true_dense[jnp.array(boundary_idx)])
t_coords_np = np.asarray(t_coords)


# ---------------------------------------------------------------------------
# 2. Per segment: honest alternate "what if" continuations, generated via
# the kernel's own transition_matrix/process_noise recursion starting from
# the EXACT true state (position and velocity) at the segment's start --
# the same math sample_prior_trajectory itself uses, just starting from a
# known state rather than drawing x_0 from the stationary prior. The true
# continuation (already computed above) is mixed in as one of the N_PATHS
# candidates at a random slot, so highlighting "the chosen one" later is
# honest -- it really is one of the shown paths, and genuinely continuous
# with its neighbors.
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

for i in range(N_COORDS - 1):
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

# ---------------------------------------------------------------------------
# scatter0: candidate initial-state draws (position marginal only -- there's
# no "previous state" to be continuous with here), with the TRUE initial
# position mixed in at a random slot, same reasoning as above.
# ---------------------------------------------------------------------------
var0 = float(kernel(t_coords[:1], t_coords[:1])[0, 0])
key, sub = jax.random.split(key)
scatter0 = np.array(jax.random.normal(sub, (N_SCATTER,)) * jnp.sqrt(var0))
key, sub = jax.random.split(key)
chosen0_idx = int(jax.random.randint(sub, (), 0, N_SCATTER))
scatter0[chosen0_idx] = y_chosen[0]

# Fixed y-axis range for the whole animation so nothing jumps around
all_vals = np.concatenate([scatter0] + [p.ravel() for p in candidate_paths])
ylim = (float(all_vals.min()) - 0.5, float(all_vals.max()) + 0.5)

# ---------------------------------------------------------------------------
# 3. Build the scene list: each entry describes what to draw for one frame.
# Rendering just clears and redraws the axes each frame from this state --
# simplest to get right for a one-off, offline render (no incremental-artist
# bookkeeping / blitting to worry about).
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
add_hold({"stage": "coords", "label": "Sample coordinates"}, "coords")
add_beat()

for frac in 0.2 + 0.8 * sweep("draw_initial_state"):
    n_shown = max(1, int(frac * N_SCATTER))
    scenes.append(
        {
            "stage": "scatter0",
            "label": "Prior distribution of possible initial states",
            "n_shown": n_shown,
        }
    )
add_hold(
    {
        "stage": "scatter0",
        "label": "Draw initial state from prior",
        "n_shown": N_SCATTER,
    },
    "initial_state_cloud",
)
add_hold(
    {"stage": "scatter0_chosen", "label": "Draw initial state from prior"},
    "initial_state_chosen",
)
add_beat()

for i in range(N_COORDS - 1):
    # The first step is where the mechanic is actually taught; the rest are
    # repetitions, so they get their own (shorter) timings.
    first = i == 0
    n_pts_total = sub_grids[i].shape[0]
    for frac in 0.2 + 0.8 * sweep("first_walk" if first else "walk"):
        n_pts = max(2, int(frac * n_pts_total))
        scenes.append(
            {
                "stage": "walk",
                "seg": i,
                "label": "Random walks to next state",
                "n_pts": n_pts,
            }
        )
    add_hold(
        {
            "stage": "walk",
            "seg": i,
            "label": "Random walks to next state",
            "n_pts": n_pts_total,
        },
        "first_walk_candidates" if first else "walk_candidates",
    )
    add_hold(
        {"stage": "walk_chosen", "seg": i, "label": "Random sample drawn"},
        "first_walk_chosen" if first else "walk_chosen",
    )
    # Longer pause after the first walk (the teaching beat) and after the last
    # one (before the paths fade away); just a breath between the repeats.
    add_beat("beat" if first or i == N_COORDS - 2 else "beat_short")

for alpha in 1.0 - sweep("fade_out"):
    scenes.append(
        {"stage": "finalize", "alpha": alpha, "label": "One realization of the process"}
    )
add_hold({"stage": "final", "label": "One realization of the process"}, "final")

N_FRAMES = len(scenes)

# ---------------------------------------------------------------------------
# 4. Render
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.5), dpi=ANIM_DPI)
# Fixed margins, set once -- NOT fig.tight_layout() per frame, which
# recomputes based on the current title's rendered size and shifts the
# axes box between frames with an empty title (stage "blank") and frames
# with real title text (every other stage).
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.22, top=0.85)


def draw_coord_lines(bold_idx=None):
    for j, tj in enumerate(t_coords_np):
        if bold_idx is not None and j == bold_idx:
            ax.axvline(tj, color="0.2", lw=2, alpha=0.85, zorder=1)
        else:
            ax.axvline(tj, color="0.5", lw=1.3, alpha=0.45, zorder=1)


def draw_chosen_chain(upto_idx):
    xs = t_coords_np[: upto_idx + 1]
    ys = y_chosen[: upto_idx + 1]
    ax.plot(xs, ys, "o", color="crimson", ms=9, zorder=10, mec="k", mew=1)


def draw_resolved_paths(upto_seg, alpha=0.6):
    """Chosen (already-decided) segments' connecting paths, drawn settled
    -- persists through later segments' animation until the final fade."""
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


def update(frame_idx):
    ax.clear()
    ax.set_xlim(-2, T_MAX + 2)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("f(t)")
    spec = scenes[frame_idx]
    stage = spec["stage"]
    ax.set_title(spec["label"], fontsize=16)

    if stage == "blank":
        pass
    elif stage == "coords":
        draw_coord_lines()
    elif stage == "scatter0":
        draw_coord_lines(bold_idx=0)
        n = spec["n_shown"]
        ax.scatter(
            np.full(n, t_coords_np[0]),
            scatter0[:n],
            color="C0",
            alpha=0.25,
            s=18,
            zorder=5,
        )
    elif stage == "scatter0_chosen":
        draw_coord_lines(bold_idx=0)
        ax.scatter(
            np.full(N_SCATTER, t_coords_np[0]),
            scatter0,
            color="C0",
            alpha=0.12,
            s=18,
            zorder=5,
        )
        draw_chosen_chain(0)
    elif stage == "walk":
        seg = spec["seg"]
        draw_coord_lines(bold_idx=seg + 1)
        draw_resolved_paths(seg)
        draw_chosen_chain(seg)
        n_pts = spec["n_pts"]
        sg = sub_grids[seg][:n_pts]
        for k in range(N_PATHS):
            ax.plot(sg, candidate_paths[seg][:n_pts, k], color="C0", alpha=0.25, lw=1)
    elif stage == "walk_chosen":
        seg = spec["seg"]
        draw_coord_lines(bold_idx=seg + 1)
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
        draw_chosen_chain(seg + 1)
    elif stage in ("finalize", "final"):
        draw_coord_lines()
        alpha = spec.get("alpha", 0.0)
        if alpha > 0:
            draw_resolved_paths(N_COORDS - 1, alpha=alpha * 0.9)
        else:
            ax.plot(
                t_coords_np, y_chosen, "-", color="crimson", lw=1.5, alpha=0.6, zorder=9
            )
        draw_chosen_chain(N_COORDS - 1)

    return []


ani = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
write_interactive_html(ani, "sample_animation.html", fps=12)
ani.save(HERE / "sample_animation.gif", writer="pillow", fps=12)
print(f"Saved sample_animation.html and .gif ({N_FRAMES} frames)")
