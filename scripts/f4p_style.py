#!/usr/bin/env python3
"""Matplotlib publication helpers for the ICDE plots, built on figures4papers.

This module implements the API that the figures4papers project specifies in
``scientific-figure-making/references/api.md`` (``FigureStyle``,
``apply_publication_style``, ``create_subplots``, ``finalize_figure``,
``make_grouped_bar``, ``annotate_bars``, ``make_trend``, ``make_scatter``) and
follows its ``common-patterns.md``: black bar edges, hatching for print-safe
separation, values printed in place on the bars, explicit y limits, and legends
that live outside the data area.

    figures4papers: https://github.com/ChenLiu-1996/figures4papers
    license: CC BY-NC 4.0; local clone: vendor/figures4papers (commit b709de1)

Four house-style defaults of figures4papers are overridden on purpose, because
the manuscript checklist fixes them:

* Times text (Liberation Serif, the metric-compatible Times New Roman clone,
  embedded as TrueType) with STIX math instead of Helvetica, so the plots match
  the body font of the paper;
* a closed four-sided axis box instead of open top and right spines;
* a light dashed horizontal grid instead of no grid;
* the Okabe-Ito colourblind-safe palette, mapped to the same semantic roles
  figures4papers uses (blue for the proposed method, grey for baselines).

Figures are drawn at their final print size, so a 10 pt label in a figure prints
at 10 pt on the page. ``layout_problems`` measures the rendered figure and reports every
overlap, so a figure with touching text never reaches the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

CM = 1.0 / 2.54
TEXT_WIDTH_CM = 18.13     # IEEEtran two-column text width (516 TeX pt)
COLUMN_WIDTH_CM = 8.85    # IEEEtran single column (252 TeX pt)

# Okabe-Ito palette, keyed by semantic role (figures4papers: "semantic colour mapping").
OKABE_ITO = {
    "blue": "#0072B2", "vermilion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#B3B3B3",
}
PALETTE = {
    "proposed": OKABE_ITO["blue"],        # ZeroClean-Loop v6
    "variant": OKABE_ITO["green"],        # ZeroClean-Loop v4
    "carry": OKABE_ITO["orange"],         # carry arms
    "adapted": OKABE_ITO["sky"],          # adapted baselines
    "baseline": OKABE_ITO["grey"],        # source ZeroED and the sequential reconstruction
    "contrast": OKABE_ITO["vermilion"],
    "highlight": OKABE_ITO["yellow"],
    "muted": "#E6E6E6",
    "ink": "#1A1A1A",
    "grid": "#CFCFCF",
}
DEFAULT_COLORS = [PALETTE[k] for k in ["proposed", "variant", "contrast", "adapted", "carry", "baseline"]]

# TrueType files embed cleanly as Type 42; the CFF-flavoured OpenType Times clones
# (Nimbus Roman) make poppler report a font-type mismatch, so they are not used.
_FONT_FILES = [
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 10.0
    axes_linewidth: float = 0.9
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Times New Roman", "Liberation Serif")
    annotation_size: float = 8.0
    boxed: bool = True
    grid: bool = True


STYLE = FigureStyle()


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Configure rcParams once, before any figure is created."""
    global STYLE
    STYLE = style or FigureStyle()
    for path in _FONT_FILES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    ink = PALETTE["ink"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in STYLE.font_family if name in installed][:1]
    if not families:
        raise RuntimeError("no Times-compatible font is installed: " + ", ".join(STYLE.font_family))
    plt.rcParams.update({
        "font.family": families,
        "font.size": STYLE.font_size,
        "mathtext.fontset": "stix",
        "text.usetex": STYLE.use_tex,
        "text.color": ink,
        "axes.labelsize": STYLE.font_size,
        "axes.titlesize": STYLE.font_size,
        "axes.linewidth": STYLE.axes_linewidth,
        "axes.edgecolor": ink,
        "axes.labelcolor": ink,
        "axes.spines.top": STYLE.boxed,
        "axes.spines.right": STYLE.boxed,
        "axes.axisbelow": True,
        "axes.grid": STYLE.grid,
        "axes.grid.axis": "y",
        "grid.color": PALETTE["grid"],
        "grid.linestyle": (0, (3.0, 2.0)),
        "grid.linewidth": 0.55,
        "xtick.labelsize": STYLE.font_size,
        "ytick.labelsize": STYLE.font_size,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
        "xtick.color": ink,
        "ytick.color": ink,
        "legend.frameon": False,
        "legend.fontsize": STYLE.font_size,
        "legend.handlelength": 1.5,
        "legend.handleheight": 0.8,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.2,
        "legend.borderpad": 0.0,
        "legend.borderaxespad": 0.0,
        "legend.labelspacing": 0.35,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "lines.markeredgewidth": 0.7,
        "lines.markeredgecolor": ink,
        "patch.linewidth": 0.7,
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "figure.dpi": 200,
    })
    return STYLE


# --------------------------------------------------------------------------- #
# Layout in centimetres at final print size
# --------------------------------------------------------------------------- #
def new_figure(width_cm: float, height_cm: float):
    return plt.figure(figsize=(width_cm * CM, height_cm * CM))


def add_axes_cm(fig, left: float, bottom: float, width: float, height: float, **kwargs):
    """Add an axes whose box is given in centimetres from the lower-left corner."""
    w, h = fig.get_size_inches() / CM
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kwargs)


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(axes).ravel()


def finalize_figure(fig, out_path, formats: Sequence[str] | None = None, dpi: int = 300,
                    close: bool = True, pad: float | None = None, **kwargs) -> list[Path]:
    """Save at the exact canvas size (``pad=None``) or tightly cropped with ``pad`` inches."""
    out_path = Path(out_path)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "svg", "eps"]
    allowed = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        if fmt not in allowed:
            raise ValueError(f"unsupported format: {fmt}")
        target = out_path.with_suffix("." + fmt)
        if pad is None:
            fig.savefig(target, dpi=dpi, **kwargs)
        else:
            fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kwargs)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def is_dark(color_hex: str, threshold: float = 128.0) -> bool:
    """Luminance test used by figures4papers to pick the annotation colour."""
    c = color_hex.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return 0.299 * r + 0.587 * g + 0.114 * b < threshold


def annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: float | None = None, padding: float = 1.6,
                  rotation: float = 90.0):
    """Print the value of every bar just above it (figures4papers in-place annotation)."""
    texts = []
    for bar in bars:
        value = bar.get_height()
        text = ax.annotate(fmt.format(value), xy=(bar.get_x() + bar.get_width() / 2, value),
                           xytext=(0, padding), textcoords="offset points", ha="center", va="bottom",
                           rotation=rotation, fontsize=fontsize or STYLE.annotation_size)
        text.set_gid("barvalue")
        text._f4p_bar = bar  # noqa: SLF001  (used by layout_problems)
        texts.append(text)
    return texts


def make_grouped_bar(ax, categories: Sequence[str], series: Sequence[Sequence[float]], labels: Sequence[str],
                     ylabel: str = "Value", colors: Sequence[str] | None = None, annotate: bool | Sequence[bool] = False,
                     hatches: Sequence[str] | None = None, group_width: float = 0.84, fmt: str = "{:.2f}"):
    """Grouped bars with black edges; returns one ``BarContainer`` per series."""
    data = np.asarray(series, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(categories):
        raise ValueError("series must be a 2D array with one column per category")
    if len(labels) != data.shape[0]:
        raise ValueError("one label per series is required")
    colors = list(colors or DEFAULT_COLORS)[: data.shape[0]]
    hatches = list(hatches or [""] * data.shape[0])
    flags = [annotate] * data.shape[0] if isinstance(annotate, bool) else list(annotate)
    x = np.arange(len(categories))
    width = group_width / data.shape[0]
    containers = []
    for k in range(data.shape[0]):
        offset = (k - (data.shape[0] - 1) / 2) * width
        bars = ax.bar(x + offset, data[k], width=width, color=colors[k], hatch=hatches[k], label=labels[k],
                      edgecolor=PALETTE["ink"], linewidth=0.7, zorder=3)
        if flags[k]:
            annotate_bars(ax, bars, fmt=fmt)
        containers.append(bars)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    return containers


def make_trend(ax, x: Sequence, y_series: Sequence[Sequence[float]], labels: Sequence[str],
               colors: Sequence[str] | None = None, ylabel: str | None = None, xlabel: str | None = None,
               show_shadow: bool = False, markers: Sequence[str] | None = None,
               linestyles: Sequence | None = None):
    """Lines with black-edged markers over categorical or numeric x positions."""
    positions = np.arange(len(x))
    colors = list(colors or DEFAULT_COLORS)
    markers = list(markers or ["o", "s", "D", "^", "v", "P"])
    linestyles = list(linestyles or ["-"] * len(y_series))
    lines = []
    for k, values in enumerate(y_series):
        values = np.asarray(values, dtype=float)
        if values.shape != positions.shape:
            raise ValueError("each series must have the same length as x")
        (line,) = ax.plot(positions, values, color=colors[k], marker=markers[k], linestyle=linestyles[k],
                          label=labels[k], zorder=4, clip_on=False)
        if show_shadow:
            ax.fill_between(positions, values - values.std(), values + values.std(), color=colors[k], alpha=0.15, lw=0)
        lines.append(line)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlim(-0.4, len(x) - 0.6)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return lines


def make_scatter(ax, x: Sequence[float], y: Sequence[float], label: str | None = None, color: str | None = None,
                 size: float = 30.0, alpha: float = 1.0, marker: str = "o"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be same-length 1D arrays")
    return ax.scatter(x, y, s=size, c=color or PALETTE["proposed"], marker=marker, alpha=alpha, label=label,
                      edgecolors=PALETTE["ink"], linewidths=0.7, zorder=5)


def legend_above(ax, handles=None, labels=None, ncol: int | None = None, pad_pt: float = 4.0,
                 span: Sequence | None = None, **kwargs):
    """Legend in a reserved band above the axes (or centred over ``span``), never over data."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    fig = ax.figure
    axes = list(span) if span else [ax]
    boxes = [a.get_position() for a in axes]
    x_mid = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    y_top = max(b.y1 for b in boxes) + pad_pt / 72.0 / fig.get_size_inches()[1]
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x_mid, y_top),
                        bbox_transform=fig.transFigure, ncol=ncol or len(handles), **kwargs)
    legend._f4p_axes = axes  # noqa: SLF001
    return legend


def subcaption(ax, text: str, offset_pt: float = 19.0, span: Sequence | None = None):
    """Sub-figure caption "(a) ..." centred below the axes (or below ``span``)."""
    fig = ax.figure
    axes = list(span) if span else [ax]
    boxes = [a.get_position() for a in axes]
    x_mid = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    y = min(b.y0 for b in boxes) - offset_pt / 72.0 / fig.get_size_inches()[1]
    return fig.text(x_mid, y, text, ha="center", va="top")


# --------------------------------------------------------------------------- #
# Measured layout check
# --------------------------------------------------------------------------- #
def _visible_tick_labels(axis) -> Iterable:
    lo, hi = sorted(axis.get_view_interval())
    span = (hi - lo) or 1.0
    for tick in axis.get_major_ticks():
        if not (lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span):
            continue
        for label in (tick.label1, tick.label2):
            if label.get_visible() and label.get_text().strip():
                yield label


def _shrunk(box: Bbox, px: float) -> Bbox:
    return Bbox.from_extents(box.x0 + px, box.y0 + px, box.x1 - px, box.y1 - px)


def _corners(box: Bbox) -> np.ndarray:
    return np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]])


def _text_polygon(artist, renderer, box: Bbox, shrink: float) -> np.ndarray:
    """Corners of the text box; slanted text gets its true oriented rectangle.

    The axis-aligned box of a rotated rectangle shares its centre, so the
    oriented rectangle is the unrotated box moved to that centre and turned.
    """
    angle = artist.get_rotation() % 180.0
    if min(angle % 90.0, 90.0 - angle % 90.0) < 1e-6:
        return _corners(_shrunk(box, shrink))
    artist.set_rotation(0.0)
    flat = artist.get_window_extent(renderer)
    artist.set_rotation(angle)
    w, h = flat.width / 2 - shrink, flat.height / 2 - shrink
    theta = np.deg2rad(angle)
    turn = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])
    return centre + np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ turn.T


def _polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (first, second):
        for k in range(len(poly)):
            edge = poly[(k + 1) % len(poly)] - poly[k]
            axis = np.array([-edge[1], edge[0]])
            a, b = first @ axis, second @ axis
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def layout_problems(fig, markers: Sequence[tuple] = (), tolerance_px: float = 0.4) -> list[str]:
    """Measure the rendered figure and list every overlap.

    Checks, in rendered pixels: (1) no two text boxes intersect; (2) every text
    lies on the canvas; (3) no legend intersects an axes box, another legend,
    or foreign text; (4) every in-axes annotation lies inside its axes; (5) a
    rotated bar value is no thicker than its bar (digits have no descenders, so
    the ink height of 0.70 em is compared); (6) no text tagged ``pointlabel``
    covers a marker passed in ``markers`` as ``(axes, x, y, radius_pt)``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72.0
    entries = []  # (description, bbox, owner)
    for ax in fig.axes:
        for artist in list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if artist.get_visible() and artist.get_text().strip():
                entries.append((artist, ax))
        for axis in (ax.xaxis, ax.yaxis):
            for label in _visible_tick_labels(axis):
                entries.append((label, ax))
    for artist in fig.texts:
        if artist.get_visible() and artist.get_text().strip():
            entries.append((artist, None))
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    for legend in legends:
        for artist in legend.get_texts():
            entries.append((artist, legend))

    problems: list[str] = []
    canvas = fig.bbox
    boxes, polygons = [], []
    for artist, owner in entries:
        box = artist.get_window_extent(renderer)
        if artist.get_gid() == "barvalue":
            # ink box of a rotated run of digits: keep the length, use 0.70 em as thickness
            ink = 0.70 * artist.get_fontsize() * scale
            mid = (box.x0 + box.x1) / 2
            box = Bbox.from_extents(mid - ink / 2, box.y0, mid + ink / 2, box.y1)
            bar = artist._f4p_bar  # noqa: SLF001
            bar_box = bar.get_window_extent(renderer)
            if ink > bar_box.width + tolerance_px:
                problems.append(f"bar value '{artist.get_text()}' is wider than its bar "
                                f"({ink / scale:.1f} pt > {bar_box.width / scale:.1f} pt)")
        boxes.append((artist, owner, box))
        polygons.append(_text_polygon(artist, renderer, box, tolerance_px))
        if box.x0 < canvas.x0 - tolerance_px or box.x1 > canvas.x1 + tolerance_px \
                or box.y0 < canvas.y0 - tolerance_px or box.y1 > canvas.y1 + tolerance_px:
            problems.append(f"text '{artist.get_text()}' leaves the canvas")
        if artist.get_gid() in {"barvalue", "inside", "pointlabel"} and owner is not None:
            frame = owner.get_window_extent(renderer)
            if box.x0 < frame.x0 - tolerance_px or box.x1 > frame.x1 + tolerance_px \
                    or box.y0 < frame.y0 - tolerance_px or box.y1 > frame.y1 + tolerance_px:
                problems.append(f"annotation '{artist.get_text()}' crosses its axes frame")
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            if _polygons_overlap(polygons[a], polygons[b]):
                problems.append(f"text '{boxes[a][0].get_text()}' overlaps text '{boxes[b][0].get_text()}'")
    legend_boxes = [(legend, legend.get_window_extent(renderer)) for legend in legends]
    for k, (legend, box) in enumerate(legend_boxes):
        inner = _shrunk(box, tolerance_px)
        for ax in fig.axes:
            if inner.overlaps(ax.get_window_extent(renderer)):
                problems.append("a legend overlaps an axes box")
        for other, other_box in legend_boxes[k + 1:]:
            if inner.overlaps(_shrunk(other_box, tolerance_px)):
                problems.append("two legends overlap")
        for artist, owner, text_box in boxes:
            if owner is not legend and inner.overlaps(_shrunk(text_box, tolerance_px)):
                problems.append(f"a legend overlaps text '{artist.get_text()}'")
        if box.x0 < canvas.x0 or box.x1 > canvas.x1 or box.y0 < canvas.y0 or box.y1 > canvas.y1:
            problems.append("a legend leaves the canvas")
    for ax, x, y, radius_pt in markers:
        cx, cy = ax.transData.transform((x, y))
        r = radius_pt * scale
        disc = Bbox.from_extents(cx - r, cy - r, cx + r, cy + r)
        for artist, _owner, box in boxes:
            if artist.get_gid() == "pointlabel" and _shrunk(box, tolerance_px).overlaps(disc):
                problems.append(f"label '{artist.get_text()}' covers a marker at ({x:.3f}, {y:.3f})")
    return problems
