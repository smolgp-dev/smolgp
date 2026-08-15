"""Shared helpers for the sample-tutorial animation scripts.

Both ``make_sample_animation.py`` and ``make_posterior_sample_animation.py``
render a scene list into an interactive HTML player (matplotlib's JSAnimation
backend), which gives play/pause, step forward/back, first/last, a frame
slider, and speed controls -- and, unlike ``ipywidgets``, needs no live Python
kernel, so it survives the static ReadTheDocs build.

Not run as part of the docs build; see either script's module docstring.
"""

import base64
import re
import uuid
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
from matplotlib.animation import HTMLWriter
from PIL import Image

# JSAnimation embeds every frame as a base64 PNG, so frame count and figure
# dpi drive the output size directly. 72 dpi keeps the rendered size sane
# while leaving the (point-based) font sizes visually unchanged.
ANIM_DPI = 72

# Outputs live beside these scripts, so they land in the right place no matter
# what directory the (manually run) generator is invoked from.
HERE = Path(__file__).parent

# These plots are flat-colour line art, so a paletted PNG is far smaller than
# matplotlib's default RGBA one at no visible cost (see _QuantizedHTMLWriter).
PALETTE_COLORS = 64

FPS = 12  # playback rate; timings below are in seconds, so this is just a grain


def seconds_to_frames(seconds: float) -> int:
    """Frame count for a duration in seconds, never rounding away to nothing."""
    return max(1, int(round(seconds * FPS)))


class Timing:
    """Named durations (in **seconds**) for an animation's key moments.

    Each script declares one of these at the top so the pacing is tunable in
    one place, in units you can actually feel, rather than as frame counts
    scattered through the storyboard. Repeated hold frames are deduplicated on
    write (see :func:`_dedupe_frames`), so lingering on an important moment
    costs playback time but almost no file size.
    """

    def __init__(self, **moments: float):
        self._moments = moments

    def __getitem__(self, moment: str) -> float:
        try:
            return self._moments[moment]
        except KeyError:
            raise KeyError(
                f"No duration defined for moment {moment!r}. Known moments: "
                f"{sorted(self._moments)}"
            ) from None

    def frames(self, moment: str) -> int:
        """Frames to spend on ``moment`` -- for holds, and for the number of
        steps in an animated transition."""
        return seconds_to_frames(self[moment])

    def total_seconds(self, counts: dict[str, int]) -> float:
        """Total runtime given how many times each moment occurs."""
        return sum(self[m] * n for m, n in counts.items())


class _QuantizedHTMLWriter(HTMLWriter):
    """:class:`~matplotlib.animation.HTMLWriter` that palette-quantizes each
    frame before base64-embedding it.

    matplotlib writes full RGBA PNGs, which is wasteful for flat-colour line
    art: quantizing to a ``PALETTE_COLORS``-entry palette shrinks these frames
    several-fold with no visible difference, which matters a lot when every
    frame is inlined into the notebook.
    """

    def grab_frame(self, **savefig_kwargs):
        if not self.embed_frames or self._hit_limit:
            return super().grab_frame(**savefig_kwargs)
        raw = BytesIO()
        self.fig.savefig(raw, format="png", dpi=self.dpi, **savefig_kwargs)
        raw.seek(0)
        quantized = (
            Image.open(raw)
            .convert("RGB")
            .quantize(colors=PALETTE_COLORS, method=Image.MEDIANCUT)
        )
        out = BytesIO()
        quantized.save(out, format="PNG", optimize=True)
        self._saved_frames.append(base64.encodebytes(out.getvalue()).decode("ascii"))


def _dedupe_frames(html: str) -> tuple[str, int, int]:
    """Collapse repeated identical frames into JS references.

    matplotlib emits one ``frames[i] = "data:image/png;base64,..."`` line per
    frame, base64-embedding every one separately -- so an N-frame hold costs N
    full copies of the same image. Holding a moment on screen is exactly how
    this animation paces its narration, so instead each repeat is rewritten as
    ``frames[i] = frames[j]``, making long holds essentially free and letting
    the pacing be tuned without watching the file size.

    Returns the rewritten HTML, the total frame count, and the unique count.
    """
    pattern = re.compile(r'( *frames\[(\d+)\] = )"(data:image/[^"]*)"', re.DOTALL)
    first_seen: dict[str, int] = {}
    total = 0

    def replace(match):
        nonlocal total
        total += 1
        prefix, index, payload = match.group(1), int(match.group(2)), match.group(3)
        if payload in first_seen:
            return f"{prefix}frames[{first_seen[payload]}]"
        first_seen[payload] = index
        return match.group(0)

    return pattern.sub(replace, html), total, len(first_seen)


def write_interactive_html(ani, out_path: str, fps: int = 12) -> None:
    """Write ``ani`` as a self-contained interactive HTML player.

    Adds autoplay on top of matplotlib's own controls: the generated player is
    paused on load, so this appends a small script that polls for the (uniquely
    named, global) animation object and starts it. Polling rather than a fixed
    delay because the object is itself constructed inside a ``setTimeout``.

    The player is wrapped in a uniquely-ided ``<div>`` so that several of these
    can coexist on one page without their controls or styles colliding.
    """
    # Mirrors Animation.to_jshtml(), but with the quantizing writer above.
    # to_jshtml() emits an embeddable fragment (not a whole page), so the
    # result drops straight into a notebook cell.
    with mpl.rc_context({"animation.embed_limit": 512}):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "anim.html")
            # dpi must be forced here rather than trusted from the figure:
            # on a HiDPI display the backend silently doubles fig.dpi (so a
            # figure asked for 72 reports 144), which would quadruple every
            # embedded frame's pixel count.
            ani.save(
                str(path),
                writer=_QuantizedHTMLWriter(
                    fps=fps, embed_frames=True, default_mode="loop"
                ),
                dpi=ANIM_DPI,
            )
            html = path.read_text()

    html, n_total, n_unique = _dedupe_frames(html)

    # matplotlib names the animation object anim<32-hex-uuid>; recover it so we
    # can start exactly this player rather than clicking every Play button on
    # the page (the tutorial embeds more than one).
    match = re.search(r"\banim([0-9a-f]{32})\b", html)
    if match is None:  # pragma: no cover -- guards against an upstream rename
        raise RuntimeError(
            "Could not find the JSAnimation object name in to_jshtml() output; "
            "matplotlib may have changed its template, so autoplay injection "
            "needs updating."
        )
    anim_name = f"anim{match.group(1)}"

    autoplay = f"""
<script>
(function() {{
    var tries = 0;
    function start() {{
        if (typeof {anim_name} !== "undefined" && {anim_name}.play_animation) {{
            {anim_name}.play_animation();
        }} else if (tries++ < 100) {{
            setTimeout(start, 100);
        }}
    }}
    start();
}})();
</script>
"""
    wrapper_id = f"smolgp-anim-{uuid.uuid4().hex[:8]}"
    with open(HERE / out_path, "w") as f:
        f.write(f'<div id="{wrapper_id}" class="smolgp-animation">\n')
        f.write(html)
        f.write(autoplay)
        f.write("\n</div>\n")
    saved = 100 * (1 - n_unique / n_total) if n_total else 0
    print(
        f"  {out_path}: {n_total} frames ({n_unique} unique, "
        f"{saved:.0f}% deduplicated)"
    )
