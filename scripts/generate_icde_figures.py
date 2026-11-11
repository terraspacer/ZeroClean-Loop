#!/usr/bin/env python3
"""Generate the ICDE manuscript figures (vector PDF + 300 DPI PNG) from frozen artifacts.

Every number and every cell value drawn in a figure is read from the frozen,
audited result artifacts (``results/**/summary_metrics.json``, prediction files,
``results/unified_loop_v6/audit/*.jsonl``) or from the benchmark CSV files.
Nothing is typed in by hand; ``paper/figures/figure_data.json`` records the
values that were drawn so they can be compared against the tables.

Run with the project environment (matplotlib lives there):

    .venv/bin/python scripts/generate_icde_figures.py [figure names]

Figures
-------
fig_example        single column   running example: one real Flights entity        (TikZ)
fig_overview       double column   input | four-stage framework + feedback | output (TikZ)
fig_trace          double column   per-cell detection and verification traces      (TikZ)
fig_benchmark      double column   parallel six-dataset comparison, 2 x 2 panels   (matplotlib)
fig_ablation       double column   component, seed, grouping, repair-policy, 2 x 2 (matplotlib)
fig_diagnostics    double column   Rayyan column profile and cell-fate accounting  (matplotlib)

The three data plots follow the figures4papers conventions
(https://github.com/ChenLiu-1996/figures4papers, CC BY-NC 4.0) through
``scripts/f4p_style.py``: black bar edges, hatching on the open-loop baseline,
values printed in place, explicit axis limits, and legends outside the data
area. That module documents the four house-style defaults that the manuscript
checklist overrides (Times text, four-sided axis box, dashed grid, Okabe-Ito).

The overview follows the system figures of HoloClean (PVLDB 2017), Baran
(PVLDB 2020), and ZeroED (ICDE 2025): an example table on each side, a framed
pipeline of named stages in between, short bullet text, and one labelled arrow
per artifact.

Design rules shared by all figures
----------------------------------
* drawn at final print size (8.85 cm single column, 18.13 cm double column), so
  text is never rescaled noticeably: 10 pt in the plots (8 pt for the values
  printed on bars and next to markers), 9 pt in the overview, and 8.5 pt in the
  running example and the traces; the manuscript body is 10 pt and its captions
  are 8 pt, so no figure text is smaller than the caption below it;
* Times text and math, matching the manuscript body;
* closed four-sided axis boxes, one dashed grid weight, inward ticks;
* one Okabe-Ito colour per method in every plot, with hatching on the open-loop
  baseline so the bars remain distinguishable in greyscale;
* legends sit in a reserved band above the axes and sub-captions (a), (b), ...
  sit in a reserved band below them; every plot must pass
  ``f4p_style.layout_problems`` (measured, no overlapping text, legend, or
  annotation) before it is written to ``paper/figures``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zeroclean.data import read_csv  # noqa: E402  (exact-string CSV reader)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f4p_style as f4p  # noqa: E402  (figures4papers conventions)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
RESULTS = ROOT / "results"
DATA = ROOT / "vendor" / "ZeroED" / "data"
TECTONIC = shutil.which("tectonic") or "tectonic"
PDFTOPPM = shutil.which("pdftoppm") or "pdftoppm"

DATASETS = ["hospital", "flights", "beers", "rayyan", "billionaire", "movies"]
SHORT = {"hospital": "Hosp", "flights": "Flig", "beers": "Beer",
         "rayyan": "Rayy", "billionaire": "Bill", "movies": "Movi"}
OURS, SRC, SEQ = "zeroclean_loop", "zeroed", "zeroed_zeroec_reconstruction"

FIGURE_DATA: dict = {}

# --------------------------------------------------------------------------- #
# Artifact access
# --------------------------------------------------------------------------- #
_cache: dict = {}


def summary(study: str) -> dict:
    if study not in _cache:
        _cache[study] = json.loads((RESULTS / study / "summary_metrics.json").read_text())
    return _cache[study]


def metric(study: str, dataset: str, method: str, task: str, key: str = "f1") -> float:
    return float(summary(study)["metrics"][dataset][method][task][key])


def macro(study: str, method: str, task: str) -> float:
    return float(summary(study)["macro_f1"][method][task])


def adapted(dataset: str, method: str, task: str) -> float:
    block = summary("unified_loop_v6")["adapted_comparison_only"]["metrics"]
    return float(block[dataset][method][task]["f1"])


def tables(dataset: str):
    cols, dirty = read_csv(DATA / f"{dataset}_error-01.csv")
    _, clean = read_csv(DATA / f"{dataset}_clean.csv")
    return cols, dirty, clean


def predictions(study: str, dataset: str, method: str) -> dict:
    path = RESULTS / study / "predictions" / dataset / f"{method}.json"
    return json.loads(path.read_text())


def audit_records(dataset: str, rows: set[int]) -> list[dict]:
    out = []
    with (RESULTS / "unified_loop_v6" / "audit" / f"{dataset}.jsonl").open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("row") in rows:
                out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# TeX helpers
# --------------------------------------------------------------------------- #
PREAMBLE = r"""\documentclass[tikz,border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{pifont}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,patterns,patterns.meta,shapes.geometric,shapes.misc}

% Okabe-Ito colourblind-safe palette (one colour per method in every figure)
\definecolor{oiBlue}{HTML}{0072B2}
\definecolor{oiVermilion}{HTML}{D55E00}
\definecolor{oiGreen}{HTML}{009E73}
\definecolor{oiOrange}{HTML}{E69F00}
\definecolor{oiPurple}{HTML}{CC79A7}
\definecolor{oiSky}{HTML}{56B4E9}
\definecolor{oiYellow}{HTML}{F0E442}
\definecolor{oiGrey}{HTML}{8C8C8C}
\definecolor{oiGrid}{HTML}{D9D9D9}
\definecolor{oiInk}{HTML}{1A1A1A}

\newcommand{\okmark}{\textcolor{oiGreen!70!black}{\ding{51}}}
\newcommand{\badmark}{\textcolor{oiVermilion}{\ding{55}}}
\newcommand{\cellfont}{\fontsize{6.5}{7.6}\selectfont}

\tikzset{
  >={Latex[length=1.7mm,width=1.4mm]},
  flow/.style={->, line width=0.6pt, draw=oiInk},
  hatch/.style={pattern={Lines[angle=45,distance=2.4pt,line width=0.35pt]}, pattern color=black!28},
}

\tikzset{subcap/.style={font=\footnotesize, anchor=north, inner sep=0pt}}
"""


def tex(value: str) -> str:
    """Escape a raw cell string for TeX and keep repeated blanks visible."""
    out = []
    previous_blank = False
    for ch in value:
        if ch == " ":
            out.append("~" if previous_blank else " ")
            previous_blank = True
            continue
        previous_blank = False
        out.append({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}.get(ch, ch))
    return "".join(out)


def cell_text(value: str) -> str:
    return r"$\varnothing$" if value == "" else tex(value)


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 2) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", r"$-$").replace("+", r"$+$")


def coords(pairs) -> str:
    return " ".join(f"({x},{num(y)})" for x, y in pairs)


# --------------------------------------------------------------------------- #
# Running example (shared by fig_example and fig_trace)
# --------------------------------------------------------------------------- #
ENTITY = "AA-404-MIA-MCO"
EXAMPLE_SOURCES = ["travelocity", "mco", "orbitz", "helloflight"]
TIME_COLS = [2, 3, 4, 5]
EXAMPLE_COLS = [2, 3, 4]  # fig_example shows three of the four time attributes so that its text can be 8.5 pt
COL_LABEL = {0: "src", 2: r"sched\_dep", 3: r"act\_dep", 4: r"sched\_arr", 5: r"act\_arr"}


def load_example() -> dict:
    cols, dirty, clean = tables("flights")
    ours = predictions("unified_loop_v6", "flights", OURS)
    src = predictions("unified_loop_v6", "flights", SRC)
    seq = predictions("unified_loop_v6", "flights", SEQ)
    ours_rep = {(r["row"], r["column"]): r["value"] for r in ours["repairs"]}
    seq_rep = {(r["row"], r["column"]): r["value"] for r in seq["repairs"]}
    entity_rows = [i for i, row in enumerate(dirty) if row[1] == ENTITY]
    by_source = {dirty[i][0]: i for i in entity_rows}
    rows = [by_source[name] for name in EXAMPLE_SOURCES]
    truth = [clean[rows[0]][j] for j in TIME_COLS]
    for i in rows:
        assert [clean[i][j] for j in TIME_COLS] == truth, "sources of one entity share one truth"
    return {
        "cols": cols, "dirty": dirty, "clean": clean, "rows": rows, "truth": truth,
        "n_sources": len(entity_rows),
        "src_flag": lambda i, j: src["probabilities"][i][j] > 0.5,
        "ours_flag": lambda i, j: ours["probabilities"][i][j] > 0.5,
        "ours_rep": ours_rep, "seq_rep": seq_rep,
        "by_source": by_source,
        "ours_prob": ours["probabilities"],
    }


def fig_example() -> str:
    ex = load_example()
    dirty, clean, rows = ex["dirty"], ex["clean"], ex["rows"]
    shown = [0] + EXAMPLE_COLS

    w_src, w_t, rh = 1.40, 2.47, 0.44  # the long open-loop edits of panel (b) fill the column at 8.5 pt
    xs = [0.0, w_src] + [w_src + w_t * k for k in range(1, len(EXAMPLE_COLS) + 1)]
    width = xs[-1]
    out: list[str] = []
    stats: dict = {}

    def cell(x0, x1, y_top, text, fill=None, hatch=False, style="", mark=None, align="left"):
        y0 = y_top - rh
        if fill:
            out.append(rf"\fill[{fill}] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y_top:.3f});")
        if hatch:
            out.append(rf"\fill[hatch] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y_top:.3f});")
        yb = y0 + 0.135  # common text baseline, so every row sits at the same height
        if align == "left":
            out.append(rf"\node[anchor=base west, inner sep=0pt] at ({x0 + 0.07:.3f},{yb:.3f}) {{\cellfont {style}{text}}};")
        else:
            out.append(rf"\node[anchor=base, inner sep=0pt] at ({(x0 + x1) / 2:.3f},{yb:.3f}) {{\cellfont {style}{text}}};")
        if mark:
            out.append(rf"\node[anchor=base east, inner sep=0pt] at ({x1 - 0.06:.3f},{yb:.3f}) {{\fontsize{{7.5}}{{8}}\selectfont {mark}}};")

    def grid(y_top, n_rows):
        y_bot = y_top - rh * n_rows
        for k in range(n_rows + 1):
            lw = "0.6pt" if k in (0, n_rows) else ("0.5pt" if k == 1 else "0.25pt")
            col = "oiInk" if k in (0, 1, n_rows) else "black!35"
            out.append(rf"\draw[line width={lw}, draw={col}] (0,{y_top - rh * k:.3f}) -- ({width:.3f},{y_top - rh * k:.3f});")
        for x in xs[1:-1]:
            out.append(rf"\draw[line width=0.25pt, draw=black!35] ({x:.3f},{y_top:.3f}) -- ({x:.3f},{y_bot:.3f});")
        out.append(rf"\draw[line width=0.6pt, draw=oiInk] (0,{y_top:.3f}) -- (0,{y_bot:.3f}) ({width:.3f},{y_top:.3f}) -- ({width:.3f},{y_bot:.3f});")
        return y_bot

    def header(y_top):
        out.append(rf"\fill[black!9] (0,{y_top - rh:.3f}) rectangle ({width:.3f},{y_top:.3f});")
        for k, j in enumerate(shown):
            cell(xs[k], xs[k + 1], y_top, COL_LABEL[j], style=r"\bfseries ")

    def title(y, label, note=""):
        out.append(rf"\node[anchor=base west, inner sep=0pt] at (0,{y + 0.13:.3f}) {{\cellfont {label}}};")
        if note:
            out.append(rf"\node[anchor=base east, inner sep=0pt] at ({width:.3f},{y + 0.13:.3f}) {{\cellfont\itshape {note}}};")

    err = r"\bfseries\color{oiVermilion}"

    # (a) dirty input -------------------------------------------------------
    y = 0.0
    n_err = sum(dirty[i][j] != clean[i][j] for i in rows for j in EXAMPLE_COLS)
    title(y, rf"\textbf{{(a)}} Dirty input $T$, flight {tex(ENTITY)}", rf"{n_err} erroneous cells")
    header(y)
    for r, i in enumerate(rows):
        yt = y - rh * (r + 1)
        cell(xs[0], xs[1], yt, tex(dirty[i][0]))
        for k, j in enumerate(EXAMPLE_COLS):
            wrong = dirty[i][j] != clean[i][j]
            cell(xs[k + 1], xs[k + 2], yt, cell_text(dirty[i][j]), style=err if wrong else "")
    yt = y - rh * (len(rows) + 1)
    out.append(rf"\fill[oiGreen!10] (0,{yt - rh:.3f}) rectangle ({width:.3f},{yt:.3f});")
    cell(xs[0], xs[1], yt, r"truth $T^{\star}$", style=r"\itshape ")
    for k, j in enumerate(EXAMPLE_COLS):
        cell(xs[k + 1], xs[k + 2], yt, tex(ex["truth"][TIME_COLS.index(j)]), style=r"\itshape ")
    y = grid(y, len(rows) + 2)

    # (b) open loop ----------------------------------------------------------
    y -= 0.60
    flagged = sum(ex["src_flag"](i, j) for i in rows for j in shown)
    edits = [(i, j) for i in rows for j in EXAMPLE_COLS if (i, j) in ex["seq_rep"]]
    good = sum(ex["seq_rep"][c] == clean[c[0]][c[1]] for c in edits)
    stats["open_loop"] = {"flagged": flagged, "shown": len(rows) * len(shown), "edits": len(edits), "correct": good}
    title(y, r"\textbf{(b)} Open-loop ZeroED $\rightarrow$ ZeroEC",
          rf"{flagged}/{len(rows) * len(shown)} cells flagged, {good}/{len(edits)} edits correct")
    header(y)
    for r, i in enumerate(rows):
        yt = y - rh * (r + 1)
        cell(xs[0], xs[1], yt, tex(dirty[i][0]), hatch=ex["src_flag"](i, 0))
        for k, j in enumerate(EXAMPLE_COLS):
            if (i, j) in ex["seq_rep"]:
                value = ex["seq_rep"][(i, j)]
                right = value == clean[i][j]
                cell(xs[k + 1], xs[k + 2], yt, tex(value),
                     fill="oiGreen!24" if right else "oiVermilion!20",
                     style="" if right else err, mark=r"\okmark" if right else r"\badmark")
            else:
                wrong = dirty[i][j] != clean[i][j]
                cell(xs[k + 1], xs[k + 2], yt, cell_text(dirty[i][j]),
                     hatch=ex["src_flag"](i, j), style=err if wrong else "")
    y = grid(y, len(rows) + 1)

    # (c) closed loop --------------------------------------------------------
    y -= 0.60
    flagged = sum(ex["ours_flag"](i, j) for i in rows for j in shown)
    flagged_true = sum(ex["ours_flag"](i, j) and dirty[i][j] != clean[i][j] for i in rows for j in shown)
    edits = [(i, j) for i in rows for j in EXAMPLE_COLS if (i, j) in ex["ours_rep"]]
    good = sum(ex["ours_rep"][c] == clean[c[0]][c[1]] for c in edits)
    missed = sum((not ex["ours_flag"](i, j)) and dirty[i][j] != clean[i][j] for i in rows for j in EXAMPLE_COLS)
    stats["closed_loop"] = {"flagged": flagged, "flagged_true": flagged_true, "edits": len(edits),
                            "correct": good, "missed": missed, "errors": n_err}
    title(y, r"\textbf{(c)} ZeroClean-Loop",
          rf"{flagged}/{len(rows) * len(shown)} cells flagged, {good}/{len(edits)} edits correct")
    header(y)
    for r, i in enumerate(rows):
        yt = y - rh * (r + 1)
        cell(xs[0], xs[1], yt, tex(dirty[i][0]), hatch=ex["ours_flag"](i, 0))
        for k, j in enumerate(EXAMPLE_COLS):
            if (i, j) in ex["ours_rep"]:
                value = ex["ours_rep"][(i, j)]
                right = value == clean[i][j]
                cell(xs[k + 1], xs[k + 2], yt, tex(value),
                     fill="oiGreen!24" if right else "oiVermilion!20",
                     style="" if right else err, mark=r"\okmark" if right else r"\badmark")
            else:
                wrong = dirty[i][j] != clean[i][j]
                cell(xs[k + 1], xs[k + 2], yt, cell_text(dirty[i][j]),
                     hatch=ex["ours_flag"](i, j), style=err if wrong else "")
    y = grid(y, len(rows) + 1)

    # legend -----------------------------------------------------------------
    y -= 0.42
    box = r"(0,-0.11) rectangle (0.40,0.11)"
    mark = r"\fontsize{7.5}{8}\selectfont"
    items = [
        (rf"\fill[hatch] {box}; \draw[line width=0.3pt] {box};", "flagged", 0.50),
        (rf"\fill[oiGreen!24] {box}; \draw[line width=0.3pt] {box}; \node[inner sep=0pt] at (0.20,0) {{{mark}\okmark}};", "correct edit", 0.50),
        (rf"\fill[oiVermilion!20] {box}; \draw[line width=0.3pt] {box}; \node[inner sep=0pt] at (0.20,0) {{{mark}\badmark}};", "wrong edit", 0.50),
        (r"\node[inner sep=0pt, anchor=base west] at (0,-0.10) {\cellfont\bfseries\color{oiVermilion}6:41};", r"differs from $T^{\star}$", 0.70),
    ]
    x = 0.0
    for (art, label, lx), w in zip(items, [1.83, 2.28, 2.19, 2.3]):
        out.append(rf"\begin{{scope}}[shift={{({x:.3f},{y:.3f})}}] {art} \node[anchor=base west, inner sep=0pt] at ({lx},-0.10) {{\cellfont {label}}}; \end{{scope}}")
        x += w

    FIGURE_DATA["fig_example"] = {
        "entity": ENTITY, "sources": EXAMPLE_SOURCES, "rows": rows,
        "entity_source_count": ex["n_sources"], "errors_shown": n_err,
        "columns": [ex["cols"][j] for j in EXAMPLE_COLS], "time_values_shown": len(rows) * len(EXAMPLE_COLS), **stats,
    }
    return PREAMBLE + "\\begin{document}\n\\renewcommand{\\cellfont}{\\fontsize{8.5}{9.6}\\selectfont}\n\\begin{tikzpicture}[x=1cm,y=1cm]\n" + "\n".join(out) + "\n\\end{tikzpicture}\n\\end{document}\n"


# --------------------------------------------------------------------------- #
# System overview (static diagram of the data flow)
# --------------------------------------------------------------------------- #
STAGES = [  # letter, tab colour, title, two body lines, artifact handed to the next stage
    ("a", "oiSky", r"Dirty-Only\\Rule Synthesis",
     [r"profile $\Pi(T)$ $\rightarrow$ LLM-written rules and FDs",
      r"AST check and sandbox run on $T$"],
     r"admitted rules $\mathcal{Q}_{\mathrm{adm}}$, FDs $\mathcal{D}_{\mathrm{adm}}$, violation matrix $V$"),
    ("b", "oiGreen", r"Fallback-Gated\\Detection",
     [r"gate $g_j$: inherited flag without a witness $\rightarrow 0.49$",
      r"witness $V, L^{\mathrm{low}}, D^{\mathrm{fd}}, G, B, K^{\mathrm{ctx}}$ $\rightarrow$ retain or $0.75$"],
     r"scores $p_{i,j}$, trigger mask $M^{\mathrm{trig}}$"),
    ("c", "oiOrange", r"Counterfactual\\Verification",
     [r"pool $|\mathcal{C}_{i,j}| \le 3$: source consensus, seed, CTS peers",
      r"substitute $v'$; recheck rules, FDs, NMI, domain"],
     r"evidence $\Delta\mathbf{f}_{i,j}(v')$, score $S_{i,j}(v')$"),
    ("d", "oiPurple", r"Guarded\\Repair",
     [r"freeze $M^{\mathrm{det}}$; select $\hat{v}_{i,j}$ with the highest $S_{i,j}$",
      r"weighted row guard: commit if $\Delta R_i^{\mathrm{row}} \ge -0.25$"],
     None),
]


def mini_table(out, x0, y_top, widths, header, rows, rh=0.38, fills=None, styles=None):
    """Small table glyph: grey header, thin inner rules, heavier frame; returns the bottom y."""
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)
    n = len(rows) + 1
    y_bot = y_top - rh * n
    out.append(rf"\fill[white] ({xs[0]:.3f},{y_bot:.3f}) rectangle ({xs[-1]:.3f},{y_top:.3f});")
    out.append(rf"\fill[black!10] ({xs[0]:.3f},{y_top - rh:.3f}) rectangle ({xs[-1]:.3f},{y_top:.3f});")
    for r, row in enumerate(rows):
        for k in range(len(row)):
            fill = (fills or {}).get((r, k))
            if fill:
                out.append(rf"\fill[{fill}] ({xs[k]:.3f},{y_top - rh * (r + 2):.3f}) rectangle ({xs[k + 1]:.3f},{y_top - rh * (r + 1):.3f});")
    for k, label in enumerate(header):
        out.append(rf"\node[anchor=base west, inner sep=0pt] at ({xs[k] + 0.08:.3f},{y_top - rh + 0.115:.3f}) {{\cellfont\bfseries {label}}};")
    for r, row in enumerate(rows):
        yb = y_top - rh * (r + 2) + 0.115
        for k, text in enumerate(row):
            style = (styles or {}).get((r, k), "")
            out.append(rf"\node[anchor=base west, inner sep=0pt] at ({xs[k] + 0.08:.3f},{yb:.3f}) {{\cellfont {style}{text}}};")
    for k in range(n + 1):
        heavy = k in (0, 1, n)
        out.append(rf"\draw[line width={'0.6pt' if heavy else '0.25pt'}, draw={'oiInk' if heavy else 'black!35'}] ({xs[0]:.3f},{y_top - rh * k:.3f}) -- ({xs[-1]:.3f},{y_top - rh * k:.3f});")
    for x in xs[1:-1]:
        out.append(rf"\draw[line width=0.25pt, draw=black!35] ({x:.3f},{y_top:.3f}) -- ({x:.3f},{y_bot:.3f});")
    out.append(rf"\draw[line width=0.6pt, draw=oiInk] ({xs[0]:.3f},{y_top:.3f}) -- ({xs[0]:.3f},{y_bot:.3f}) ({xs[-1]:.3f},{y_top:.3f}) -- ({xs[-1]:.3f},{y_bot:.3f});")
    return y_bot


def fig_overview() -> str:
    """Input | framework with four stacked stages and the feedback loop | output.

    The layout follows the overview figures of HoloClean, Baran, and ZeroED: example
    tables on both sides, a framed pipeline of named stages in between, short bullet
    text, and one labelled arrow per artifact. The two side tables show one real
    column of the running example (read from the artifacts, not typed in).
    """
    ex = load_example()
    dirty, rows = ex["dirty"], ex["rows"]
    col = 4  # sched_arr, the column of the cell traced in fig_trace
    width = f4p.TEXT_WIDTH_CM
    lx1 = 3.00                               # left column: inputs
    fx0, fx1 = 3.53, 13.98                   # framework frame
    mx0, mx1, tab = fx0 + 0.16, fx1 - 0.52, 2.70  # stage boxes and their coloured tab
    rx0 = width - 3.70                       # right column: LLM, evaluator, output
    mh, mg = 1.10, 0.52                      # stage height and the gap that holds the artifact arrow
    ys = {"d": 0.18 + mh / 2}
    for prev, key in zip("dcb", "cba"):
        ys[key] = ys[prev] + mh + mg
    f_top = ys["a"] + mh / 2 + 0.56

    out = [r"\renewcommand{\cellfont}{\fontsize{8.5}{9.6}\selectfont}",
           r"\begin{tikzpicture}[x=1cm,y=1cm, font=\fontsize{9}{10.6}\selectfont,",
           r"  pin/.style={draw=oiInk, line width=0.6pt, densely dashed, rounded corners=2pt, fill=white, align=center, inner sep=2pt, minimum height=0.92cm},",
           r"  block/.style={line width=2.6pt, draw=black!62, -{Triangle[length=2.7mm,width=4.0mm]}},",
           r"  art/.style={->, line width=1.0pt, draw=oiInk}]"]

    # framework frame and title
    out.append(rf"\draw[rounded corners=5pt, line width=0.8pt, draw=oiInk, fill=black!5] ({fx0},0) rectangle ({fx1},{f_top:.3f});")
    out.append(rf"\node[anchor=west, inner sep=0pt] at ({mx0 + 0.02:.3f},{f_top - 0.29:.3f}) {{\textbf{{ZeroClean-Loop}}: every stage reads only $T$ and the pinned inputs}};")

    # the four stages
    for letter, colour, title, body, _ in STAGES:
        y = ys[letter]
        y0, y1 = y - mh / 2, y + mh / 2
        out.append(r"\begin{scope}")
        out.append(rf"\clip[rounded corners=3pt] ({mx0:.3f},{y0:.3f}) rectangle ({mx1:.3f},{y1:.3f});")
        out.append(rf"\fill[white] ({mx0:.3f},{y0:.3f}) rectangle ({mx1:.3f},{y1:.3f});")
        out.append(rf"\fill[{colour}!36] ({mx0:.3f},{y0:.3f}) rectangle ({mx0 + tab:.3f},{y1:.3f});")
        out.append(r"\end{scope}")
        out.append(rf"\draw[rounded corners=3pt, line width=0.7pt, draw=oiInk] ({mx0:.3f},{y0:.3f}) rectangle ({mx1:.3f},{y1:.3f});")
        out.append(rf"\draw[line width=0.5pt, draw=oiInk] ({mx0 + tab:.3f},{y0:.3f}) -- ({mx0 + tab:.3f},{y1:.3f});")
        out.append(rf"\node[anchor=west, inner sep=0pt, font=\fontsize{{9}}{{10.6}}\selectfont\bfseries] at ({mx0 + 0.10:.3f},{y:.3f}) {{({letter})}};")
        out.append(rf"\node[anchor=west, inner sep=0pt, align=left, font=\fontsize{{9}}{{10.6}}\selectfont\bfseries] at ({mx0 + 0.56:.3f},{y:.3f}) {{{title}}};")
        for line, dy in zip(body, (0.215, -0.215)):
            out.append(rf"\node[anchor=west, inner sep=0pt] at ({mx0 + tab + 0.13:.3f},{y + dy:.3f}) {{\textbullet~{line}}};")
        out.append(rf"\coordinate ({letter}W) at ({mx0:.3f},{y:.3f}); \coordinate ({letter}E) at ({mx1:.3f},{y:.3f});")

    # artifacts handed from stage to stage
    xa = mx0 + 0.95
    for (letter, _, _, _, artifact), nxt in zip(STAGES, "bcd"):
        y_from, y_to = ys[letter] - mh / 2, ys[nxt] + mh / 2
        out.append(rf"\draw[art] ({xa:.3f},{y_from:.3f}) -- ({xa:.3f},{y_to:.3f});")
        out.append(rf"\node[anchor=west, inner sep=0pt] at ({xa + 0.20:.3f},{(y_from + y_to) / 2:.3f}) {{{artifact}}};")

    # feedback: repair-side evidence returns to detection before the mask is frozen
    xf = mx1 + 0.32
    out.append(rf"\draw[->, line width=1.4pt, draw=oiVermilion, rounded corners=3pt] ({mx1:.3f},{ys['c'] + 0.12:.3f}) -- ({xf:.3f},{ys['c'] + 0.12:.3f}) -- ({xf:.3f},{ys['b'] - 0.12:.3f}) -- ({mx1 + 0.02:.3f},{ys['b'] - 0.12:.3f});")
    out.append(rf"\node[anchor=east, inner sep=0pt, text=oiVermilion!88!black] at ({mx1 - 0.10:.3f},{(ys['b'] + ys['c']) / 2:.3f}) {{feedback: $K^{{\mathrm{{ctx}}}}$, promote, demote}};")

    # left column: the dirty table and the pinned inherited inputs
    rh = 0.38
    src_w = 1.38
    t_top = ys["a"] + rh * 2.5
    in_rows = [[tex(dirty[i][0]), cell_text(dirty[i][col])] for i in rows]
    mini_table(out, 0.0, t_top, [src_w, lx1 - src_w], ["src", COL_LABEL[col]], in_rows, rh=rh)
    out.append(rf"\node[anchor=south, inner sep=0pt] at ({lx1 / 2:.3f},{t_top + 0.09:.3f}) {{\textbf{{Input}}: dirty table $T$}};")
    out.append(rf"\draw[block] ({lx1 + 0.07:.3f},{ys['a']:.3f}) -- ({fx0 - 0.03:.3f},{ys['a']:.3f});")
    pins = [("b", r"ZeroED scores\\$P^{(0)}$, $P^{\mathrm{emp}}$, and $H$"),
            ("c", r"seed repairs $\mathcal{U}_{\mathrm{seed}}$,\\cached CTS pools")]
    for letter, text in pins:
        out.append(rf"\node[pin, minimum width={lx1 - 0.03:.2f}cm] (pin{letter}) at ({lx1 / 2:.3f},{ys[letter]:.3f}) {{{text}}};")
        out.append(rf"\draw[art] (pin{letter}.east) -- ({letter}W);")
    out.append(rf"\node[inner sep=0pt, align=center, font=\fontsize{{9}}{{10.6}}\selectfont\itshape] at ({lx1 / 2:.3f},{ys['d']:.3f}) {{all inputs pinned by\\SHA-256 before use}};")

    # right column: LLM rule author, evaluator, and the output
    xc = (rx0 + width) / 2
    out.append(rf"\node[pin, minimum width={width - rx0 - 0.03:.2f}cm] (llm) at ({xc:.3f},{ys['a']:.3f}) {{LLM rule author\\profile out, rules in}};")
    out.append(rf"\draw[art, <->] (aE) -- (llm.west);")
    flags = [int(ex["ours_flag"](i, col)) for i in rows]
    values = [ex["ours_rep"].get((i, col), dirty[i][col]) for i in rows]
    out_rows = [[tex(dirty[i][0]), cell_text(v), str(m)] for i, v, m in zip(rows, values, flags)]
    fills = {(r, 1): "oiBlue!20" for r, i in enumerate(rows) if (i, col) in ex["ours_rep"]}
    o_top = rh * 5
    mask_w = 0.84
    mini_table(out, rx0, o_top, [src_w, width - rx0 - src_w - mask_w, mask_w],
               ["src", COL_LABEL[col], r"$M^{\mathrm{det}}$"], out_rows, rh=rh, fills=fills)
    out.append(rf"\node[anchor=south, inner sep=0pt] (olab) at ({xc:.3f},{o_top + 0.09:.3f}) {{\textbf{{Output}}: repaired table $\hat{{T}}$}};")
    out.append(rf"\draw[block] ({mx1 + 0.05:.3f},{ys['d']:.3f}) -- ({rx0 - 0.07:.3f},{ys['d']:.3f});")
    y_eval = o_top + 0.09 + 0.33 + 0.42 + 0.46
    out.append(rf"\node[pin, fill=black!10, minimum width={width - rx0 - 0.03:.2f}cm] (eval) at ({xc:.3f},{y_eval:.3f}) {{evaluator only: $T^{{\star}}$\\exact-string metrics}};")
    out.append(rf"\draw[->, line width=0.8pt, densely dashed, draw=oiInk] ({xc:.3f},{o_top + 0.09 + 0.33:.3f}) -- (eval.south);")
    out.append(r"\end{tikzpicture}")

    FIGURE_DATA["fig_overview"] = {
        "column": col, "rows": rows,
        "input": [dirty[i][col] for i in rows], "output": values, "mask": flags,
        "edited": [(i, col) in ex["ours_rep"] for i in rows],
    }
    return PREAMBLE + "\\begin{document}\n" + "\n".join(out) + "\n\\end{document}\n"


# --------------------------------------------------------------------------- #
# Per-cell traces of the running example
# --------------------------------------------------------------------------- #
WITNESS = {
    "validated_target_specific_rule_violation": r"$V$",
    "dirty_leave_one_out_low_support": r"$L^{\mathrm{low}}$",
    "low_dirty_leave_one_out_value_support": r"$L^{\mathrm{low}}$",
    "low_dirty_leave_one_out_format_support": r"$L^{\mathrm{low}}$",
    "independent_source_consensus_contradiction": r"$K^{\mathrm{ctx}}$",
    "low_dirty_entity_peer_support": r"$K^{\mathrm{ctx}}$",
    "intrinsic_dirty_fd": r"$D^{\mathrm{fd}}$",
}
ORDER = [r"$V$", r"$L^{\mathrm{low}}$", r"$D^{\mathrm{fd}}$", r"$K^{\mathrm{ctx}}$"]


def detection_trace(ex: dict, row: int, col: int, records: list[dict]) -> dict:
    recs = [r for r in records if r["event"] == "detection" and r["row"] == row and r["column"] == col]
    assert recs, (row, col)
    witnesses: list[str] = []
    for rec in recs:
        for kind in rec.get("evidence_types") or []:
            label = WITNESS.get(kind)
            if label and label not in witnesses:
                witnesses.append(label)
    witnesses.sort(key=ORDER.index)
    actions = [r["action"] for r in recs]
    final = ex["ours_prob"][row][col]
    inherited = recs[0]["inherited_score"]
    if any(a.startswith("promote") for a in actions):
        action = "promote"
    elif any(a.startswith("demote") for a in actions):
        action = "demote"
    else:
        action = "retain"
    dirty, clean = ex["dirty"], ex["clean"]
    return {
        "row": row, "column": col, "source": dirty[row][0], "column_name": COL_LABEL[col],
        "value": dirty[row][col], "truth": clean[row][col], "is_error": dirty[row][col] != clean[row][col],
        "inherited": inherited, "final": final, "witnesses": witnesses, "action": action,
        "mask": int(final > 0.5),
    }


def fig_trace() -> str:
    ex = load_example()
    dirty, clean = ex["dirty"], ex["clean"]
    src_rows = {name: ex["by_source"][name] for name in ["travelocity", "mco", "helloflight"]}
    iad_row = next(i for i, row in enumerate(dirty) if row[0] == "iad" and row[2] == "12:30")
    records = audit_records("flights", set(src_rows.values()) | {iad_row})
    traces = [
        detection_trace(ex, src_rows["travelocity"], 4, records),
        detection_trace(ex, src_rows["mco"], 4, records),
        detection_trace(ex, iad_row, 2, records),
        detection_trace(ex, src_rows["helloflight"], 3, records),
    ]
    diag = json.loads((RESULTS / "unified_loop_v6" / "diagnostics" / "flights.json").read_text())
    gate = {c["column_name"]: c for c in diag["detection"]["columns"]}
    assert all(gate[n]["gated"] and gate[n]["source_positive_fraction"] == 1.0
               for n in ["sched_dep_time", "act_dep_time", "sched_arr_time"])

    out = [r"\renewcommand{\cellfont}{\fontsize{8.5}{9.7}\selectfont}",
           r"\begin{tikzpicture}[x=1cm,y=1cm, font=\cellfont,",
           r"  flow/.style={->, line width=0.8pt, draw=oiInk},",
           r"  subcap/.style={font=\fontsize{9}{10.6}\selectfont, anchor=north, inner sep=0pt},",
           r"  chip/.style={draw=oiInk, line width=0.55pt, rounded corners=2pt, inner sep=1.6pt, align=center, fill=white, minimum height=0.76cm},",
           r"  head/.style={font=\cellfont\bfseries, inner sep=0pt, align=center}]"]

    # ---------------- (a) detection trace ----------------------------------
    bands = [(0.00, 2.90, "dirty cell"), (3.04, 4.32, r"inherited\\$p^{\mathrm{emp}}_{i,j}$"),
             (4.46, 6.30, r"gate $g_j = 1$,\\witnesses"),
             (6.44, 7.26, r"final\\$p_{i,j}$"), (7.40, 8.30, r"mask\\$M^{\mathrm{det}}_{i,j}$")]
    mid = [(x0 + x1) / 2 for x0, x1, _ in bands]
    top, y_rows = 5.30, [4.06, 3.11, 2.16, 1.21]
    for x0, x1, label in bands:
        out.append(rf"\fill[black!6, rounded corners=2pt] ({x0:.2f},0.70) rectangle ({x1:.2f},{top:.2f});")
        out.append(rf"\node[head] at ({(x0 + x1) / 2:.2f},{top - 0.36:.2f}) {{{label}}};")
        out.append(rf"\draw[line width=0.4pt, draw=black!45] ({x0 + 0.06:.2f},{top - 0.73:.2f}) -- ({x1 - 0.06:.2f},{top - 0.73:.2f});")
    detection_data = []
    for y, t in zip(y_rows, traces):
        value = cell_text(t["value"])
        vstyle = r"\bfseries\color{oiVermilion}" if t["is_error"] else ""
        out.append(rf"\node[chip, minimum width=2.80cm] (c) at ({mid[0]:.3f},{y}) {{{tex(t['source'])} $\cdot$ {t['column_name']}\\{{{vstyle}{value}}}}};")
        out.append(rf"\node[chip, minimum width=0.86cm] (i) at ({mid[1]:.3f},{y}) {{{t['inherited']:.2f}}};")
        if t["witnesses"]:
            label = ", ".join(t["witnesses"])
            verb = "promote" if t["action"] == "promote" else "retain"
            out.append(rf"\node[chip, minimum width=1.72cm, fill=oiSky!28] (w) at ({mid[2]:.3f},{y}) {{{label}\\{verb}}};")
        else:
            out.append(rf"\node[chip, minimum width=1.72cm, fill=black!10] (w) at ({mid[2]:.3f},{y}) {{no witness\\demote}};")
        fill = "oiBlue!22" if t["mask"] else "white"
        out.append(rf"\node[chip, minimum width=0.72cm, fill={fill}] (f) at ({mid[3]:.3f},{y}) {{{t['final']:.2f}}};")
        correct = bool(t["mask"]) == t["is_error"]
        mark = r"\okmark" if correct else r"\badmark"
        out.append(rf"\node[chip, minimum width=0.78cm, fill={fill}] (m) at ({mid[4]:.3f},{y}) {{{t['mask']}~{mark}}};")
        out.append(r"\draw[flow] (c) -- (i); \draw[flow] (i) -- (w); \draw[flow] (w) -- (f); \draw[flow] (f) -- (m);")
        detection_data.append({k: t[k] for k in ["row", "column", "source", "value", "truth", "inherited", "final", "witnesses", "action", "mask"]})
    out.append(r"\node[anchor=west, inner sep=0pt] at (0,0.40) {All columns gated ($\gamma_j = 1.00 > \tau_{\mathrm{src}}$); marks compare $M^{\mathrm{det}}$ with $T^{\star}$};")
    out.append(r"\node[subcap] at (4.15,0.06) {(a) Detection: four real cells through the gate};")

    # ---------------- (b) verification trace --------------------------------
    row = src_rows["mco"]
    col = 4
    sel = next(r for r in records if r["event"] == "repair_selection" and r["row"] == row and r["column"] == col)
    guard = next(r for r in records if r["event"] == "combined_row_guard" and r["row"] == row)
    reviews = sorted(sel["candidate_reviews"], key=lambda c: -c["weighted_delta"])
    weights = diag["repair"]["config"]["feature_weights"]
    assert sel["selected_value"] == reviews[0]["value"] and not guard["rolled_back"]
    for review in reviews:
        recomputed = sum(w * d for w, d in zip(weights, review["delta_features"]))
        assert abs(recomputed - review["weighted_delta"]) < 1e-9

    ox = 8.68
    bw_total = f4p.TEXT_WIDTH_CM - ox
    out.append(rf"\begin{{scope}}[shift={{({ox},0)}}]")
    # tuple strip
    tw, tx0, ty = 1.52, 3.37, top
    out.append(rf"\node[anchor=west, inner sep=0pt, align=left] at (0,{ty - 0.56:.2f}) {{tuple $r_i$ of source {tex(dirty[row][0])},\\flight {tex(ENTITY)}}};")
    for k, j in enumerate(TIME_COLS):
        x0 = tx0 + k * tw
        out.append(rf"\node[inner sep=0pt, font=\cellfont\bfseries] at ({x0 + tw / 2:.2f},{ty - 0.17:.2f}) {{{COL_LABEL[j]}}};")
        target = j == col
        fill = "oiBlue!16" if target else "white"
        out.append(rf"\draw[line width={'0.9pt' if target else '0.45pt'}, draw={'oiBlue' if target else 'oiInk'}, fill={fill}] ({x0 + 0.03:.2f},{ty - 0.86:.2f}) rectangle ({x0 + tw - 0.03:.2f},{ty - 0.38:.2f});")
        wrong = dirty[row][j] != clean[row][j]
        style = r"\bfseries\color{oiVermilion}" if wrong else ""
        out.append(rf"\node[inner sep=0pt] at ({x0 + tw / 2:.2f},{ty - 0.62:.2f}) {{{style}{cell_text(dirty[row][j])}}};")
    xt = tx0 + TIME_COLS.index(col) * tw + tw / 2

    # candidate table
    cx = [0.00, 1.95, 4.05, 4.83, 5.61, 6.39, 7.17, 8.22, bw_total]
    heads = [r"candidate $v'$", "origin", r"$\Delta r$", r"$\Delta f$", r"$\Delta n$", r"$\Delta d$", r"$S_{i,j}(v')$", "gate"]
    th, rh = 3.94, 0.50
    out.append(rf"\fill[black!9] (0,{th - rh:.2f}) rectangle ({bw_total:.2f},{th:.2f});")
    for k, label in enumerate(heads):
        out.append(rf"\node[head] at ({(cx[k] + cx[k + 1]) / 2:.2f},{th - rh / 2:.2f}) {{{label}}};")
    out.append(rf"\draw[flow, draw=oiBlue] ({xt:.2f},{ty - 0.86:.2f}) -- ({xt:.2f},{th + 0.02:.2f});")
    out.append(rf"\node[anchor=east, inner sep=0pt] at ({xt - 0.12:.2f},{(ty - 0.86 + th) / 2:.2f}) {{substitute each $v'$ for the target cell}};")
    verification_data = []
    for r, review in enumerate(reviews):
        yt = th - rh * (r + 1)
        ym = yt - rh / 2
        selected = review["value"] == sel["selected_value"]
        if selected:
            out.append(rf"\fill[oiGreen!14] (0,{yt - rh:.2f}) rectangle ({bw_total:.2f},{yt:.2f});")
        value_style = r"\bfseries " if selected else ""
        out.append(rf"\node[anchor=west, inner sep=0pt] at (0.07,{ym:.2f}) {{{value_style}{tex(review['value'])}}};")
        origins = set(review["origins"])
        if "key_source_consensus" in origins:
            cons = review["consensus"]
            origin = rf"consensus {cons['winning_source_count']}/{cons['peer_source_count']}"
        elif "seed" in origins:
            origin = "open-loop seed"
        else:
            origin = "CTS peers"
        out.append(rf"\node[anchor=west, inner sep=0pt] at ({cx[1] + 0.07:.2f},{ym:.2f}) {{{origin}}};")
        for k, delta in enumerate(review["delta_features"]):
            x0, x1 = cx[2 + k], cx[3 + k]
            if abs(delta) >= 0.005:
                colour = "oiGreen" if delta > 0 else "oiVermilion"
                out.append(rf"\fill[{colour}!{int(12 + 45 * min(1.0, abs(delta)))}] ({x0 + 0.04:.2f},{yt - rh + 0.05:.2f}) rectangle ({x1 - 0.04:.2f},{yt - 0.05:.2f});")
            out.append(rf"\node[inner sep=0pt] at ({(x0 + x1) / 2:.2f},{ym:.2f}) {{{signed(delta)}}};")
        out.append(rf"\node[inner sep=0pt] at ({(cx[6] + cx[7]) / 2:.2f},{ym:.2f}) {{{value_style}{signed(review['weighted_delta'], 3)}}};")
        if selected:
            gate_label = r"\okmark~select"
        elif review["eligible"]:
            gate_label = "eligible"
        else:
            gate_label = r"\badmark~format"
            assert "invalid_format" in review["rejection_reasons"]
        out.append(rf"\node[inner sep=0pt] at ({(cx[7] + cx[8]) / 2:.2f},{ym:.2f}) {{{gate_label}}};")
        verification_data.append({"value": review["value"], "origins": sorted(origins),
                                  "delta_features": review["delta_features"],
                                  "score": review["weighted_delta"], "eligible": review["eligible"],
                                  "selected": selected, "consensus": review.get("consensus")})
    n_rows = len(reviews) + 1
    for k in range(n_rows + 1):
        lw = "0.6pt" if k in (0, 1, n_rows) else "0.25pt"
        out.append(rf"\draw[line width={lw}, draw={'oiInk' if k in (0, 1, n_rows) else 'black!35'}] (0,{th - rh * k:.2f}) -- ({bw_total:.2f},{th - rh * k:.2f});")
    out.append(rf"\draw[line width=0.6pt, draw=oiInk] (0,{th:.2f}) -- (0,{th - rh * n_rows:.2f}) ({bw_total:.2f},{th:.2f}) -- ({bw_total:.2f},{th - rh * n_rows:.2f});")
    yb = th - rh * n_rows
    wtxt = ", ".join(f"{w:.2f}" for w in weights)
    out.append(rf"\node[anchor=west, inner sep=0pt] at (0,{yb - 0.27:.2f}) {{$S_{{i,j}}(v') = \mathbf{{w}}^{{\top}}\Delta\mathbf{{f}}_{{i,j}}(v')$, $\mathbf{{w}} = ({wtxt})$; eligible: $S_{{i,j}} \ge -0.50$}};")
    # commit chain
    yc = 0.98
    truth = clean[row][col]
    match = sel["selected_value"] == truth
    out.append(rf"\node[chip, minimum width=2.10cm, fill=oiGreen!14] (s) at (1.08,{yc}) {{selected $\hat{{v}}_{{i,j}}$\\{{\bfseries {tex(sel['selected_value'])}}}}};")
    out.append(rf"\node[chip, minimum width=3.60cm] (g) at (4.62,{yc}) {{row guard, {len(guard['columns'])} edited cells\\$\Delta R_i^{{\mathrm{{row}}}} = {guard['weighted_rule_delta']:+.2f} \ge -0.25$}};")
    out.append(rf"\node[chip, minimum width=2.20cm, fill=oiBlue!22] (k) at ({bw_total - 1.11:.2f},{yc}) {{commit to $\hat{{T}}$\\{'equals' if match else 'differs from'} truth {r'\okmark' if match else r'\badmark'}}};")
    out.append(r"\draw[flow] (s) -- (g); \draw[flow] (g) -- (k);")
    out.append(rf"\node[subcap] at ({bw_total / 2:.2f},0.06) {{(b) Verification: the three candidates of one flagged cell}};")
    out.append(r"\end{scope}")
    out.append(r"\end{tikzpicture}")

    FIGURE_DATA["fig_trace"] = {
        "detection": detection_data,
        "verification": {"row": row, "column": col, "original": dirty[row][col], "truth": truth,
                         "candidates": verification_data, "selected": sel["selected_value"],
                         "row_guard": {k: guard[k] for k in ["columns", "weighted_rule_delta", "rolled_back", "new_rule_violation"]},
                         "feature_weights": weights},
    }
    return PREAMBLE + "\\begin{document}\n" + "\n".join(out) + "\n\\end{document}\n"


# --------------------------------------------------------------------------- #
# Data plots (matplotlib, figures4papers conventions; see scripts/f4p_style.py)
# --------------------------------------------------------------------------- #
INK = f4p.PALETTE["ink"]
METHOD = {  # one colour and one hatch per method in every plot
    "src": dict(color=f4p.PALETTE["baseline"], hatch="////"),
    "adp": dict(color=f4p.PALETTE["adapted"], hatch=""),
    "carry": dict(color=f4p.PALETTE["carry"], hatch=""),
    "v4": dict(color=f4p.PALETTE["variant"], hatch=""),
    "v6": dict(color=f4p.PALETTE["proposed"], hatch=""),
}
F1_ED, F1_EC = r"$F_1^{\mathrm{ED}}$", r"$F_1^{\mathrm{EC}}$"
MACRO_ED, MACRO_EC = r"$\bar{F}_1^{\mathrm{ED}}$", r"$\bar{F}_1^{\mathrm{EC}}$"
UNIT_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# shared geometry of the two-by-two result figures (centimetres)
LEFT, PANEL_W, PANEL_GAP, PANEL_H = 1.45, 7.45, 1.63, 3.30
ROW_GAP, BOTTOM, TOP_BAND = 2.30, 1.72, 0.78
SUBCAP, SUBCAP_XLABEL = 21, 33  # offset of a sub-caption below the axes without / with an x label (pt)


def two_by_two():
    height = BOTTOM + PANEL_H + ROW_GAP + PANEL_H + TOP_BAND
    fig = f4p.new_figure(f4p.TEXT_WIDTH_CM, height)
    y_low, y_high = BOTTOM, BOTTOM + PANEL_H + ROW_GAP
    x_left, x_right = LEFT, LEFT + PANEL_W + PANEL_GAP
    axes = [f4p.add_axes_cm(fig, x, y, PANEL_W, PANEL_H)
            for x, y in [(x_left, y_high), (x_right, y_high), (x_left, y_low), (x_right, y_low)]]
    return fig, axes


def bars(ax, categories, series, keys, labels, ylabel, annotate=False, ymax=1.0):
    containers = f4p.make_grouped_bar(
        ax, categories, [series[k] for k in keys], labels, ylabel=ylabel,
        colors=[METHOD[k]["color"] for k in keys], hatches=[METHOD[k]["hatch"] for k in keys], annotate=annotate)
    ax.set_ylim(0.0, ymax)
    ax.set_yticks(UNIT_TICKS)
    return containers


def benchmark_data() -> dict:
    v6, v4, seed = "unified_loop_v6", "unified_loop_v4", "conservative_seed_v1"
    ed = {
        "src": [metric(v6, d, SRC, "detection") for d in DATASETS],
        "adp": [adapted(d, "adapted_zeroed", "detection") for d in DATASETS],
        "carry": [metric(seed, d, "empty_carry", "detection") for d in DATASETS],
        "v4": [metric(v4, d, OURS, "detection") for d in DATASETS],
        "v6": [metric(v6, d, OURS, "detection") for d in DATASETS],
    }
    ec = {
        "src": [metric(v6, d, SEQ, "repair") for d in DATASETS],
        "adp": [adapted(d, "adapted_zeroed_zeroec", "repair") for d in DATASETS],
        "carry": [metric(seed, d, "seeded_carry", "repair") for d in DATASETS],
        "v4": [metric(v4, d, OURS, "repair") for d in DATASETS],
        "v6": [metric(v6, d, OURS, "repair") for d in DATASETS],
    }
    pr = {
        "src": [(metric(v6, d, SRC, "detection", "recall"), metric(v6, d, SRC, "detection", "precision")) for d in DATASETS],
        "v4": [(metric(v4, d, OURS, "detection", "recall"), metric(v4, d, OURS, "detection", "precision")) for d in DATASETS],
        "v6": [(metric(v6, d, OURS, "detection", "recall"), metric(v6, d, OURS, "detection", "precision")) for d in DATASETS],
    }
    versions = ["v2", "v3", "v4", "v5", "v6"]
    traj = {
        "ed": [macro(f"unified_loop_{v}", OURS, "detection") for v in versions],
        "ec": [macro(f"unified_loop_{v}", OURS, "repair") for v in versions],
        "rayyan": [metric(f"unified_loop_{v}", "rayyan", OURS, "detection") for v in versions],
    }
    FIGURE_DATA["fig_benchmark"] = {"datasets": DATASETS, "detection_f1": ed, "repair_f1": ec,
                                    "recall_precision": pr, "versions": versions, "trajectory": traj}
    return {"ed": ed, "ec": ec, "pr": pr, "versions": versions, "traj": traj}


def fig_benchmark():
    data = benchmark_data()
    names = [SHORT[d] for d in DATASETS]
    keys = ["src", "adp", "carry", "v4", "v6"]
    labels = ["Source", "Adapted", "Carry", "v4", "v6"]
    fig, (ax_a, ax_b, ax_c, ax_d) = two_by_two()
    markers = []

    # (a), (b): the proposed method carries its value; Table II lists every bar
    flags = [False, False, False, False, True]
    bars(ax_a, names, data["ed"], keys, labels, F1_ED, annotate=flags, ymax=1.12)
    bars(ax_b, names, data["ec"], keys, labels, F1_EC, annotate=flags, ymax=1.12)
    handles, texts = ax_a.get_legend_handles_labels()
    f4p.legend_above(ax_a, handles, texts, span=[ax_a, ax_b], columnspacing=2.0)

    # (c) operating points: one arrow per dataset from source ZeroED to v6
    pr = data["pr"]
    for a, b in zip(pr["src"], pr["v6"]):
        ax_c.annotate("", xy=b, xytext=a, zorder=3,
                      arrowprops=dict(arrowstyle="-|>", color="#6E6E6E", lw=1.0, shrinkA=5.0, shrinkB=5.0,
                                      mutation_scale=10))
    style = {"src": ("^", f4p.PALETTE["baseline"], 62, "Source"), "v4": ("s", f4p.PALETTE["variant"], 52, "v4"),
             "v6": ("o", f4p.PALETTE["proposed"], 54, "v6")}
    for key in ["src", "v4", "v6"]:
        marker, colour, size, label = style[key]
        xs, ys = zip(*pr[key])
        f4p.make_scatter(ax_c, xs, ys, label=label, color=colour, size=size, marker=marker)
        markers += [(ax_c, x, y, 3.9) for x, y in pr[key]]
    # dataset name next to the source marker (next to the v6 marker for Billionaire, whose
    # source marker lies inside the cluster of repaired points)
    place = {"hospital": ("src", (5, -6), "left", "top"), "flights": ("src", (7, -1), "left", "center"),
             "beers": ("src", (7, -1), "left", "center"), "rayyan": ("src", (0, -7), "center", "top"),
             "billionaire": ("v6", (7, -1), "left", "center"), "movies": ("src", (-7, 0), "right", "center")}
    for k, d in enumerate(DATASETS):
        which, offset, ha, va = place[d]
        text = ax_c.annotate(SHORT[d], xy=pr[which][k], xytext=offset, textcoords="offset points", ha=ha, va=va,
                             fontsize=f4p.STYLE.annotation_size + 0.5)
        text.set_gid("pointlabel")
    ax_c.set_xlim(0.0, 1.14)
    ax_c.set_ylim(-0.06, 1.06)
    ax_c.set_xticks(UNIT_TICKS)
    ax_c.set_yticks(UNIT_TICKS)
    ax_c.grid(True, axis="x")
    ax_c.set_xlabel(r"$R^{\mathrm{ED}}$", labelpad=1.5)
    ax_c.set_ylabel(r"$P^{\mathrm{ED}}$")
    f4p.legend_above(ax_c, columnspacing=2.0, handletextpad=0.1)

    # (d) cohorts v2 to v6
    traj = data["traj"]
    f4p.make_trend(ax_d, data["versions"], [traj["ed"], traj["ec"], traj["rayyan"]],
                   [MACRO_ED, MACRO_EC, "Rayy " + F1_ED],
                   colors=[f4p.PALETTE["proposed"], f4p.PALETTE["contrast"], f4p.PALETTE["variant"]],
                   markers=["o", "s", "D"], linestyles=["-", "-", (0, (4, 2))], ylabel=r"$F_1$")
    ax_d.set_ylim(0.2, 0.9)
    ax_d.set_yticks([0.2, 0.4, 0.6, 0.8])
    f4p.legend_above(ax_d, columnspacing=2.0)

    f4p.subcaption(ax_a, "(a) Detection " + F1_ED + " per dataset", SUBCAP)
    f4p.subcaption(ax_b, "(b) Repair " + F1_EC + " per dataset", SUBCAP)
    f4p.subcaption(ax_c, "(c) Detection operating points", SUBCAP_XLABEL)
    f4p.subcaption(ax_d, "(d) Cohorts v2 to v6", SUBCAP_XLABEL)
    return fig, markers


def ablation_data() -> dict:
    comp, seed, v6, v4, fl = ("components_source_v2", "conservative_seed_v1", "unified_loop_v6",
                              "unified_loop_v4", "flight_model_spec_v1")
    a = {
        "without_multi_column_first_pass": [metric(comp, d, "without_multi_column_first_pass", "detection") for d in DATASETS],
        "without_counterfactual_first_pass": [metric(comp, d, "without_counterfactual_first_pass", "detection") for d in DATASETS],
        "full_first_verified_round": [metric(comp, d, "full_first_verified_round", "detection") for d in DATASETS],
    }
    b = {
        "seq": [macro(seed, SEQ, "detection"), macro(seed, SEQ, "repair"),
                metric(seed, "flights", SEQ, "detection"), metric(seed, "billionaire", SEQ, "repair")],
        "empty": [macro(seed, "empty_carry", "detection"), macro(seed, "empty_carry", "repair"),
                  metric(seed, "flights", "empty_carry", "detection"), metric(seed, "billionaire", "empty_carry", "repair")],
        "seeded": [macro(seed, "seeded_carry", "detection"), macro(seed, "seeded_carry", "repair"),
                   metric(seed, "flights", "seeded_carry", "detection"), metric(seed, "billionaire", "seeded_carry", "repair")],
        "v6": [macro(v6, OURS, "detection"), macro(v6, OURS, "repair"),
               metric(v6, "flights", OURS, "detection"), metric(v6, "billionaire", OURS, "repair")],
    }
    specs = ["lambda0p0", "lambda0p5", "lambda0p9", "selectedEB", "equal"]
    c = {
        "attribute_near": [metric(fl, "flights", f"attribute_near_{k}", "detection") for k in specs],
        "profile_complete": [metric(fl, "flights", f"profile_complete_{k}", "detection") for k in specs],
        "v6": metric(v6, "flights", OURS, "detection"),
    }
    keys = ["precision", "recall", "f1"]
    d = {
        "seq": [metric(v6, ds, SEQ, "repair", k) for ds in ["billionaire", "beers"] for k in keys],
        "v4": [metric(v4, ds, OURS, "repair", k) for ds in ["billionaire", "beers"] for k in keys],
        "v6": [metric(v6, ds, OURS, "repair", k) for ds in ["billionaire", "beers"] for k in keys],
    }
    FIGURE_DATA["fig_ablation"] = {
        "component_detection_f1": a,
        "seed_arms": {"groups": ["macro ED", "macro EC", "flights ED", "billionaire EC"], **b},
        "flights_grouping": {"specs": specs, **c},
        "repair_policy": {"order": "billionaire P,R,F1 then beers P,R,F1", **d}}
    return {"a": a, "b": b, "c": c, "d": d}


def fig_ablation():
    data = ablation_data()
    names = [SHORT[d] for d in DATASETS]
    fig, (ax_a, ax_b, ax_c, ax_d) = two_by_two()
    markers = []

    # (a) first-pass components
    a = data["a"]
    f4p.make_grouped_bar(
        ax_a, names, [a["without_multi_column_first_pass"], a["without_counterfactual_first_pass"],
                      a["full_first_verified_round"]],
        ["w/o MC", "w/o CF", "full"], ylabel=F1_ED,
        colors=[METHOD["carry"]["color"], METHOD["adp"]["color"], METHOD["v6"]["color"]], annotate=True,
        group_width=0.78)
    ax_a.set_ylim(0.0, 1.10)
    ax_a.set_yticks(UNIT_TICKS)
    f4p.legend_above(ax_a, columnspacing=2.0)

    # (b) seed-carry arms
    b = data["b"]
    f4p.make_grouped_bar(
        ax_b, [MACRO_ED, MACRO_EC, "Flig " + F1_ED, "Bill " + F1_EC],
        [b["seq"], b["empty"], b["seeded"], b["v6"]], ["Seq.", "Empty", "Seeded", "v6"], ylabel=r"$F_1$",
        colors=[METHOD[k]["color"] for k in ["src", "adp", "carry", "v6"]],
        hatches=[METHOD[k]["hatch"] for k in ["src", "adp", "carry", "v6"]], annotate=True, group_width=0.74)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_yticks(UNIT_TICKS)
    f4p.legend_above(ax_b, columnspacing=1.3)

    # (c) Flights source grouping
    c = data["c"]
    ticks = [r"$\lambda = 0$", r"$\lambda = 0.5$", r"$\lambda = 0.9$", "EB", "Equal"]
    f4p.make_trend(ax_c, ticks, [c["attribute_near"], c["profile_complete"]], ["Attr.", "Profile"],
                   colors=[f4p.PALETTE["carry"], f4p.PALETTE["variant"]], markers=["s", "o"],
                   ylabel="Flights " + F1_ED)
    ax_c.axhline(c["v6"], color=f4p.PALETTE["proposed"], linestyle=(0, (4, 2)), linewidth=2.0, label="v6", zorder=2)
    for k in range(len(ticks)):
        for key, dy, va in [("attribute_near", -7, "top"), ("profile_complete", 7, "bottom")]:
            # the last profile point sits on the v6 reference line, so its value goes below it
            if key == "profile_complete" and k == len(ticks) - 1:
                dy, va = -7, "top"
            text = ax_c.annotate(f"{c[key][k]:.3f}", xy=(k, c[key][k]), xytext=(0, dy), textcoords="offset points",
                                 ha="center", va=va, fontsize=f4p.STYLE.annotation_size)
            text.set_gid("pointlabel")
            markers.append((ax_c, k, c[key][k], 3.4))
    ax_c.set_ylim(0.66, 0.80)
    ax_c.set_yticks([0.68, 0.72, 0.76, 0.80])
    f4p.legend_above(ax_c, columnspacing=2.0)

    # (d) repair policy
    d = data["d"]
    f4p.make_grouped_bar(
        ax_d, [r"$P^{\mathrm{EC}}$", r"$R^{\mathrm{EC}}$", F1_EC] * 2, [d["seq"], d["v4"], d["v6"]],
        ["Seq.", "v4", "v6"], ylabel="EC metric",
        colors=[METHOD[k]["color"] for k in ["src", "v4", "v6"]],
        hatches=[METHOD[k]["hatch"] for k in ["src", "v4", "v6"]], annotate=True, group_width=0.78)
    ax_d.set_ylim(0.0, 1.30)
    ax_d.set_yticks(UNIT_TICKS)
    ax_d.axvline(2.5, color="#6E6E6E", linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    for x, label in [(1.0, "Billionaire"), (4.0, "Beers")]:
        ax_d.annotate(label, xy=(x, 0.0), xycoords=("data", "axes fraction"), xytext=(0, -18),
                      textcoords="offset points", ha="center", va="top")
    f4p.legend_above(ax_d, columnspacing=2.0)

    f4p.subcaption(ax_a, "(a) First-pass components", SUBCAP)
    f4p.subcaption(ax_b, "(b) Seed-carry arms", SUBCAP)
    f4p.subcaption(ax_c, "(c) Flights source grouping", SUBCAP_XLABEL)
    f4p.subcaption(ax_d, "(d) Repair policy, v4 vs. v6", SUBCAP_XLABEL)
    return fig, markers


# --------------------------------------------------------------------------- #
# Diagnostics: Rayyan column profile and cell-fate accounting
# --------------------------------------------------------------------------- #
RAYYAN_LABEL = {
    "id": "id", "article_title": "title", "article_language": "lang", "journal_title": "jtitle",
    "journal_abbreviation": "jabbr", "journal_issn": "issn", "article_jvolumn": "vol",
    "article_jissue": "issue", "article_jcreated_at": "date", "article_pagination": "pages",
    "author_list": "auth",
}


def cell_fates(dataset: str) -> dict:
    cols, dirty, clean = tables(dataset)
    ours = predictions("unified_loop_v6", dataset, OURS)
    src = predictions("unified_loop_v6", dataset, SRC)
    repairs = {(r["row"], r["column"]): r["value"] for r in ours["repairs"]}
    fate = {"kept_error": 0, "kept_clean": 0, "dropped_clean": 0, "dropped_error": 0, "new_error": 0, "new_clean": 0}
    flagged = 0
    for i, row in enumerate(dirty):
        for j in range(len(cols)):
            is_error = row[j] != clean[i][j]
            s = src["probabilities"][i][j] > 0.5
            z = ours["probabilities"][i][j] > 0.5
            flagged += z
            if s:
                fate[("kept_" if z else "dropped_") + ("error" if is_error else "clean")] += 1
            elif z:
                fate["new_error" if is_error else "new_clean"] += 1
    correct = sum(clean[i][j] == value for (i, j), value in repairs.items())
    det = summary("unified_loop_v6")["metrics"][dataset][OURS]["detection"]
    rep = summary("unified_loop_v6")["metrics"][dataset][OURS]["repair"]
    assert fate["kept_error"] + fate["new_error"] == det["tp"], dataset
    assert fate["kept_clean"] + fate["new_clean"] == det["fp"], dataset
    assert correct == rep["tp"] and len(repairs) - correct == rep["fp"], dataset
    assert all(ours["probabilities"][i][j] > 0.5 for i, j in repairs), dataset
    return {"source_flagged": sum(fate[k] for k in ["kept_error", "kept_clean", "dropped_clean", "dropped_error"]),
            "flagged": flagged, **fate, "correct_edit": correct, "wrong_edit": len(repairs) - correct,
            "no_edit": flagged - len(repairs)}


def diagnostics_data() -> dict:
    cols, dirty, _ = tables("rayyan")
    n = len(dirty)
    coverage = [sum(1 for row in dirty if row[j] != "") / n for j in range(len(cols))]
    counts = {}
    for key, study, method in [("src", "unified_loop_v6", SRC), ("v4", "unified_loop_v4", OURS), ("v6", "unified_loop_v6", OURS)]:
        probs = predictions(study, "rayyan", method)["probabilities"]
        counts[key] = [sum(probs[i][j] > 0.5 for i in range(n)) for j in range(len(cols))]
    fates = {d: cell_fates(d) for d in DATASETS}
    FIGURE_DATA["fig_diagnostics"] = {"rayyan_columns": list(cols), "rayyan_coverage": coverage,
                                      "rayyan_flagged": counts, "cell_fates": fates}
    return {"cols": cols, "coverage": coverage, "counts": counts, "fates": fates}


FATE_STYLE = {
    "good": dict(color=f4p.PALETTE["proposed"], hatch=""),
    "spurious": dict(color=f4p.PALETTE["highlight"], hatch=""),
    "bad": dict(color=f4p.PALETTE["contrast"], hatch="\\\\\\\\"),
    "none": dict(color=f4p.PALETTE["muted"], hatch=""),
}


def stacked_shares(ax, names, parts, min_label=11.0):
    """Horizontal 100 % stacked bars with the share printed inside every wide segment."""
    y = list(range(len(names)))
    left = [0.0] * len(names)
    for style_key, values in parts:
        style = FATE_STYLE[style_key]
        ax.barh(y, values, left=left, height=0.66, color=style["color"], hatch=style["hatch"],
                edgecolor=INK, linewidth=0.7, zorder=3)
        for k, value in enumerate(values):
            boxed = bool(style["hatch"])
            # a boxed label hides the hatch, so hatched segments need more room
            if value >= (min_label + 4.0 if boxed else min_label):
                colour = INK if boxed or not f4p.is_dark(style["color"]) else "white"
                text = ax.text(left[k] + value / 2, y[k], f"{value:.0f}", ha="center", va="center_baseline",
                               fontsize=f4p.STYLE.annotation_size, color=colour, zorder=5,
                               bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none") if boxed else None)
                text.set_gid("inside")
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(len(names) - 0.45, -0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.grid(True, axis="x")


def fig_diagnostics():
    data = diagnostics_data()
    cols, coverage, counts, fates = data["cols"], data["coverage"], data["counts"], data["fates"]
    names = [SHORT[d] for d in DATASETS]
    panel_h, bottom, band = 3.40, 1.95, 1.22
    fig = f4p.new_figure(f4p.TEXT_WIDTH_CM, bottom + panel_h + band)
    ax_a = f4p.add_axes_cm(fig, 1.62, bottom, 6.05, panel_h)
    ax_b = f4p.add_axes_cm(fig, 10.05, bottom, 3.50, panel_h)
    ax_c = f4p.add_axes_cm(fig, 14.17, bottom, 3.50, panel_h)

    # (a) Rayyan: flags per column (left axis) and non-empty coverage (right axis)
    labels = [RAYYAN_LABEL[c] for c in cols]
    f4p.make_grouped_bar(ax_a, labels, [counts["src"], counts["v4"], counts["v6"]], ["Source", "v4", "v6"],
                         ylabel=r"flagged cells $N_{\mathrm{pos}}$",
                         colors=[METHOD[k]["color"] for k in ["src", "v4", "v6"]],
                         hatches=[METHOD[k]["hatch"] for k in ["src", "v4", "v6"]], group_width=0.80)
    ax_a.set_ylim(0, 1200)
    ax_a.set_yticks([0, 250, 500, 750, 1000])
    # eleven column names do not fit side by side at 10 pt, so they are slanted
    plt.setp(ax_a.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax_a.tick_params(axis="x", pad=2.0)
    twin = ax_a.twinx()
    twin.grid(False)
    x = list(range(len(cols)))
    (cov_line,) = twin.plot(x, coverage, color=f4p.PALETTE["contrast"], marker="o", markersize=5.6, linewidth=1.8,
                            label=r"$\eta_j$", zorder=6)
    gate_line = twin.axhline(0.80, color=f4p.PALETTE["contrast"], linestyle=(0, (4, 2)), linewidth=1.4,
                             label="gate 0.80", zorder=2)
    twin.set_ylim(0, 1.2)
    twin.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    twin.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    twin.set_ylabel(r"coverage $\eta_j$")
    handles, texts = ax_a.get_legend_handles_labels()
    f4p.legend_above(ax_a, handles + [cov_line, gate_line], texts + [r"$\eta_j$", "gate 0.80"], columnspacing=1.5)

    # (b), (c): fate of every flag as a share of the flags
    def share(keys, denom):
        return {k: [100.0 * fates[d][k] / fates[d][denom] for d in DATASETS] for k in keys}

    flags = share(["kept_error", "kept_clean", "dropped_error", "dropped_clean"], "source_flagged")
    edits = share(["correct_edit", "wrong_edit", "no_edit"], "flagged")
    stacked_shares(ax_b, names, [("good", flags["kept_error"]), ("spurious", flags["kept_clean"]),
                                 ("bad", flags["dropped_error"]), ("none", flags["dropped_clean"])])
    stacked_shares(ax_c, names, [("good", edits["correct_edit"]), ("bad", edits["wrong_edit"]),
                                 ("none", edits["no_edit"])])
    ax_c.set_yticklabels([])
    for ax in (ax_b, ax_c):
        ax.set_xlabel(r"share of flags (%)", labelpad=2.0)
    patches = [Patch(facecolor=FATE_STYLE[k]["color"], hatch=FATE_STYLE[k]["hatch"], edgecolor=INK, linewidth=0.7)
               for k in ["good", "bad", "spurious", "none"]]
    f4p.legend_above(ax_b, patches, ["error kept / correct edit", "error lost / wrong edit",
                                     "clean kept", "clean lost / no edit"],
                     ncol=2, span=[ax_b, ax_c], columnspacing=1.0, handlelength=1.3)

    f4p.subcaption(ax_a, "(a) Rayyan: flagged cells and coverage per column", 38)
    f4p.subcaption(ax_b, "(b) Fate of Source flags", 38)
    f4p.subcaption(ax_c, "(c) Fate of v6 flags", 38)
    return fig, []


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
DIAGRAMS = {"fig_example": fig_example, "fig_overview": fig_overview, "fig_trace": fig_trace}
PLOTS = {"fig_benchmark": fig_benchmark, "fig_ablation": fig_ablation, "fig_diagnostics": fig_diagnostics}
FIGURES = {**DIAGRAMS, **PLOTS}


def report_size(name: str) -> None:
    size = subprocess.run(["/usr/bin/pdfinfo", str(FIG_DIR / f"{name}.pdf")], capture_output=True, text=True).stdout
    page = next(line for line in size.splitlines() if line.startswith("Page size"))
    w_pt, h_pt = float(page.split()[2]), float(page.split()[4])
    print(f"{name}: {w_pt / 28.3465:.2f} cm x {h_pt / 28.3465:.2f} cm")


def build_diagram(name: str) -> None:
    tex_path = FIG_DIR / f"{name}.tex"
    tex_path.write_text(DIAGRAMS[name](), encoding="utf-8")
    result = subprocess.run([TECTONIC, "-o", str(FIG_DIR), str(tex_path)], capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(line for line in result.stderr.splitlines() if "Requested font" not in line and " -> " not in line)
        raise SystemExit(f"{name}: TeX failed\n{tail[-3000:]}")
    subprocess.run([PDFTOPPM, "-png", "-r", "300", "-singlefile", str(FIG_DIR / f"{name}.pdf"), str(FIG_DIR / name)], check=True)


def build_plot(name: str) -> None:
    fig, markers = PLOTS[name]()
    problems = f4p.layout_problems(fig, markers)
    if problems:
        f4p.finalize_figure(fig, Path("/tmp") / f"{name}_rejected", formats=["png"], dpi=200)
        raise SystemExit(f"{name}: layout check failed (see /tmp/{name}_rejected.png)\n  " + "\n  ".join(problems))
    f4p.finalize_figure(fig, FIG_DIR / name, formats=["pdf", "png"], dpi=300)
    print(f"{name}: layout check passed (no overlapping text, legends, or annotations)")


def build(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (build_diagram if name in DIAGRAMS else build_plot)(name)
    report_size(name)


def main() -> None:
    f4p.apply_publication_style()
    selected = sys.argv[1:] or list(FIGURES)
    for name in selected:
        build(name)
    data_path = FIG_DIR / "figure_data.json"
    existing = json.loads(data_path.read_text()) if data_path.exists() else {}
    existing.update(FIGURE_DATA)
    data_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
