"""Draw the book's concept figures.

    uv run python scripts/make_figures.py            # all figures
    uv run python scripts/make_figures.py rope-rotation attention-masks

Each figure is drawn twice, into book/docs/assets/figures/NAME.svg (light theme)
and NAME-dark.svg (dark theme). In the Markdown, reference only NAME.svg; the
hook book/hooks/figures.py adds the dark variant. Figures that show numbers
compute them here (attention weights, IK solutions, scaling-law curves), so
they always agree with the formulas in the text.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "book/docs/assets/figures"

HUES = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]
THEMES = {
    "light": dict(surface="#ffffff", ink="#1b1b1b", muted="#5f5e5a", faint="#d6d5d0",
                  hex=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]),
    "dark": dict(surface="#1e2029", ink="#e8e8e6", muted="#a9a8a0", faint="#454852",
                 hex=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2f9e2f", "#9085e9", "#e66767"]),
}


def mix(a, b, t):
    """t * colour a + (1 - t) * colour b."""
    return tuple(t * x + (1 - t) * y for x, y in zip(to_rgb(a), to_rgb(b)))


class Theme:
    def __init__(self, name):
        t = THEMES[name]
        self.name, self.surface, self.ink, self.muted, self.faint = name, t["surface"], t["ink"], t["muted"], t["faint"]
        self.hue = dict(zip(HUES, t["hex"]))
        self.dark = name == "dark"

    def __getitem__(self, hue):
        return self.hue[hue]

    def soft(self, hue, t=None):
        return mix(self.hue[hue], self.surface, t if t is not None else (0.38 if self.dark else 0.18))

    def seq(self, hue="blue"):  # magnitude: one hue, surface -> saturated
        return LinearSegmentedColormap.from_list("seq", [mix(self.faint, self.surface, 0.35), self.hue[hue]])

    def div(self):  # sign: blue - neutral - orange
        return LinearSegmentedColormap.from_list("div", [self.hue["blue"], mix(self.faint, self.surface, 0.6), self.hue["orange"]])

    def on(self, rgb):
        """Readable text colour on top of fill `rgb`."""
        r, g, b = to_rgb(rgb)[:3]
        return "#111111" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"


def style(C):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "cm",
        "text.color": C.ink, "axes.labelcolor": C.ink, "axes.edgecolor": C.muted, "axes.titlecolor": C.ink,
        "xtick.color": C.muted, "ytick.color": C.muted, "xtick.labelcolor": C.ink, "ytick.labelcolor": C.ink,
        "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": C.faint, "grid.linewidth": 0.6,
        "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "axes.prop_cycle": cycler(color=[C[h] for h in HUES]), "svg.hashsalt": "vla-course",
        "image.interpolation": "nearest",
    })


FIGS = {}


def figure(fn):
    FIGS[fn.__name__.replace("_", "-")] = fn
    return fn


# ---------------------------------------------------------------- drawing helpers
# Schematic figures use a canvas whose data coordinates are inches, so text sizes
# (points) and box sizes can be reasoned about together: 10 pt text ≈ 0.08 in/char.

def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, C, x, y, w, h, text="", hue="blue", fs=10, fill=True, ls="-", r=0.06, lw=1.2, color=None, **kw):
    """Rounded box with lower-left corner (x, y)."""
    fc = (color if color is not None else C.soft(hue)) if fill else "none"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=C[hue] if hue else C.muted, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=kw.pop("tc", C.ink), **kw)


def arrow(ax, C, p, q, color=None, lw=1.3, ls="-", rad=0.0, head=True, ms=11):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>" if head else "-", color=color or C.muted, lw=lw, linestyle=ls,
        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}", mutation_scale=ms))


def chip_w(text, fs):
    return 0.068 * len(text) * fs / 10 + 0.14


def chips(ax, C, x, y, texts, hues, h=0.3, fs=9, gap=0.05, fill=True, ls="-", mono=False, widths=None):
    """A row of token chips starting at (x, y). Returns the chips' centre x positions."""
    if isinstance(hues, str):
        hues = [hues] * len(texts)
    centres = []
    for i, (t, hu) in enumerate(zip(texts, hues)):
        w = widths if widths is not None else chip_w(t, fs)
        box(ax, C, x, y, w, h, t, hu, fs=fs, fill=fill, ls=ls, r=0.05, family="monospace" if mono else None)
        centres.append(x + w / 2)
        x += w + gap
    return centres


def cells(ax, C, x, y, rows, cols, s, hue="blue", fills=None, texts=None, fs=7, ls="-", lw=0.8, tc=None, edge=None):
    """Grid of square cells, top-left corner at (x, y)."""
    for r in range(rows):
        for c in range(cols):
            fc = fills[r][c] if fills is not None else C.soft(hue)
            ax.add_patch(Rectangle((x + c * s, y - (r + 1) * s), s, s, fc=fc, ec=edge or C[hue], lw=lw, ls=ls))
            if texts is not None and texts[r][c] != "":
                col = tc or (C.on(fc) if fills is not None and fc != "none" else C.ink)
                ax.text(x + (c + .5) * s, y - (r + .5) * s, texts[r][c], ha="center", va="center", fontsize=fs, color=col)


def heat(ax, C, M, cmap, vmin, vmax, rows=None, cols=None, annot=None, title=None, fs=7):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(M.shape[1]), cols if cols is not None else [""] * M.shape[1], fontsize=8, rotation=90 if cols else 0)
    ax.set_yticks(range(M.shape[0]), rows if rows is not None else [""] * M.shape[0], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    if annot:
        norm = plt.Normalize(vmin, vmax)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, annot.format(v), ha="center", va="center", fontsize=fs, color=C.on(cmap(norm(v))))


def label(ax, C, x, y, text, fs=9, color=None, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, text, fontsize=fs, color=color or C.muted, **kw)


# ======================================================================= Chapter 0

@figure
def tensor_ranks(C):
    fig, ax = canvas(8, 2.3)
    top = 1.95

    def name(x, k, shape, note=""):
        label(ax, C, x, 0.42, f"rank {k}", fs=10, color=C.ink)
        label(ax, C, x, 0.18, shape + note, fs=9, family="monospace")

    cells(ax, C, 0.35, top - 0.45, 1, 1, 0.34, "blue", texts=[["3.1"]], fs=8)
    name(0.52, 0, "()")
    cells(ax, C, 1.2, top - 0.45, 1, 5, 0.26, "blue", texts=[["0.2", "-1", "0.7", "3", "0.1"]], fs=6.5)
    name(1.85, 1, "(d,)")
    cells(ax, C, 3.0, top - 0.05, 4, 5, 0.24, "blue")
    label(ax, C, 3.6, top + 0.08, "n", fs=9)
    label(ax, C, 2.9, top - 0.53, "m", fs=9)
    name(3.6, 2, "(m, n)")
    # one RGB image (3, H, W): three channel planes
    for k, hue in enumerate(["blue", "green", "red"]):
        cells(ax, C, 4.55 + 0.12 * (2 - k), top - 0.12 * (2 - k) - 0.05, 4, 4, 0.2, hue)
    name(5.1, 3, "(3, H, W)", "  an image")
    # a batch of images (B, 3, H, W)
    for b in range(3):
        for k, hue in enumerate(["blue", "green", "red"]):
            cells(ax, C, 6.15 + 0.62 * b + 0.07 * (2 - k), top - 0.07 * (2 - k) - 0.25, 3, 3, 0.15, hue, lw=0.6)
    name(7.0, 4, "(B, 3, H, W)", "  a batch")
    return fig


@figure
def broadcasting(C):
    fig, ax = canvas(7.6, 2.5)
    T, S, s, top = 4, 5, 0.3, 1.85
    # (T, 1): one real column, the rest virtual
    cells(ax, C, 0.3, top, T, 1, s, "blue", texts=[[str(i)] for i in range(T)], fs=8)
    cells(ax, C, 0.3 + s, top, T, S - 1, s, "blue", fills=[["none"] * (S - 1)] * T,
          texts=[[str(i)] * (S - 1) for i in range(T)], fs=8, ls=(0, (2, 2)), tc=C.muted)
    label(ax, C, 0.3 + S * s / 2, top + 0.22, "a: shape (T, 1) = (4, 1)", color=C.ink)
    label(ax, C, 2.15, top - T * s / 2, "−", fs=16, color=C.ink)
    cells(ax, C, 2.5, top, 1, S, s, "orange", texts=[[str(j) for j in range(S)]], fs=8)
    cells(ax, C, 2.5, top - s, T - 1, S, s, "orange", fills=[["none"] * S] * (T - 1),
          texts=[[str(j) for j in range(S)]] * (T - 1), fs=8, ls=(0, (2, 2)), tc=C.muted)
    label(ax, C, 2.5 + S * s / 2, top + 0.22, "b: shape (1, S) = (1, 5)", color=C.ink)
    label(ax, C, 4.35, top - T * s / 2, "=", fs=16, color=C.ink)
    D = np.subtract.outer(np.arange(T), np.arange(S))
    cmap, norm = C.div(), plt.Normalize(-4, 4)
    cells(ax, C, 4.7, top, T, S, s, "violet", fills=[[cmap(norm(v)) for v in r] for r in D],
          texts=[[str(v) for v in r] for r in D], fs=8, edge=C.surface)
    label(ax, C, 4.7 + S * s / 2, top + 0.22, "a − b: shape (T, S) = (4, 5)", color=C.ink)
    label(ax, C, 3.6, 0.32, "dashed cells are virtual: the size-1 dimension is repeated without copying memory", fs=9)
    return fig


@figure
def mlp_diagram(C):
    fig, ax = canvas(7.6, 3.3)
    xs = [0.9, 2.9, 4.9, 6.9]
    sizes = [2, 6, 6, 2]
    names = [["$p_x$", "$p_y$"], None, None, ["$q_1$", "$q_2$"]]
    hues = ["blue", "aqua", "aqua", "orange"]
    pos = []
    for x, n in zip(xs, sizes):
        ys = np.linspace(1.85 - 0.33 * (n - 1) / 2, 1.85 + 0.33 * (n - 1) / 2, n)
        pos.append([(x, y) for y in ys])
    for a, b in zip(pos, pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=C.faint, lw=0.7, zorder=0)
    for k, (layer, hue) in enumerate(zip(pos, hues)):
        for i, (x, y) in enumerate(layer):
            ax.add_patch(Circle((x, y), 0.12, fc=C.soft(hue, 0.45 if C.dark else 0.3), ec=C[hue], lw=1.2, zorder=2))
            if names[k]:
                label(ax, C, x + (-0.38 if k == 0 else 0.38), y, names[k][::-1][i], fs=11, color=C.ink)
        if sizes[k] == 6:
            label(ax, C, layer[0][0], 1.85 - 0.33 * 2.5 - 0.3, "64 units\n(6 drawn)", fs=8)
    label(ax, C, xs[0], 3.05, "input $\\mathbf{x}$\nend-effector position", fs=9, color=C.ink)
    label(ax, C, xs[3], 3.05, "output $f_\\theta(\\mathbf{x})$\njoint angles", fs=9, color=C.ink)
    label(ax, C, xs[1], 3.05, "hidden $\\mathbf{h}_1$", fs=10, color=C.ink)
    label(ax, C, xs[2], 3.05, "hidden $\\mathbf{h}_2$", fs=10, color=C.ink)
    for k in range(3):
        mid = (xs[k] + xs[k + 1]) / 2
        txt = f"$W_{k + 1}, \\mathbf{{b}}_{k + 1}$" + ("\nthen $\\sigma$ = ReLU" if k < 2 else "\n(no activation)")
        label(ax, C, mid, 0.32, txt, fs=9.5, color=C.ink)
    return fig


@figure
def relu_pieces(C):
    x = np.linspace(-3, 3, 600)
    g = np.sin(1.6 * x) + 0.25 * x

    def fit(m):
        knots = np.linspace(-3, 3, m + 2)[1:-1]
        Phi = np.column_stack([np.ones_like(x), x] + [np.maximum(0, x - k) for k in knots])
        c, *_ = np.linalg.lstsq(Phi, g, rcond=None)
        return knots, c, Phi @ c, Phi

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.28))
    knots, c, approx, Phi = fit(4)
    for j, k in enumerate(knots):
        a.plot(x, c[j + 2] * Phi[:, j + 2], lw=1.6, color=C[HUES[j]])
    a.plot(x, c[0] + c[1] * x, lw=1.6, color=C.muted, ls="--")
    a.set_title("4 units $c_j\\,\\mathrm{ReLU}(x-b_j)$; dashed: affine term")
    a.axhline(0, color=C.faint, lw=0.8, zorder=0)
    a.set_xlabel("$x$")
    b.plot(x, g, color=C.muted, lw=3, alpha=0.5, label="target $g(x)$")
    b.plot(x, approx, color=C["orange"], lw=1.8, label="sum of 4 units")
    _, _, approx16, _ = fit(16)
    b.plot(x, approx16, color=C["blue"], lw=1.6, label="sum of 16 units")
    b.plot(knots, np.interp(knots, x, approx), "o", ms=5, color=C["orange"])
    b.set_title("their sum: one kink per unit")
    b.legend(loc="lower right")
    b.set_xlabel("$x$")
    return fig


@figure
def ik_averaging(C):
    l1, l2 = 1.0, 0.8
    target = np.array([1.15, 0.85])
    r2 = target @ target
    sols = []
    for sgn in (+1, -1):
        q2 = sgn * math.acos((r2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
        q1 = math.atan2(target[1], target[0]) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        sols.append((q1, q2))
    mean = tuple(np.mean(sols, axis=0))

    def joints(q):
        e = np.array([l1 * math.cos(q[0]), l1 * math.sin(q[0])])
        return np.array([[0, 0], e, e + [l2 * math.cos(q[0] + q[1]), l2 * math.sin(q[0] + q[1])]])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for q, hue, txt, ls in [(sols[1], "blue", "demo A: elbow up", "-"), (sols[0], "aqua", "demo B: elbow down", "-"),
                            (mean, "red", "MSE prediction:\naverage of A and B", "--")]:
        J = joints(q)
        ax.plot(J[:, 0], J[:, 1], ls, color=C[hue], lw=3, solid_capstyle="round")
        ax.plot(J[:2, 0], J[:2, 1], "o", color=C[hue], ms=8, mfc=C.surface, mew=2)
        ax.plot(*J[2], "o", color=C[hue], ms=5)
        at = {"blue": (-0.5, 1.25), "aqua": (1.25, -0.05), "red": (1.42, 1.3)}[hue]
        ax.text(*at, txt, color=C[hue], fontsize=9)
    miss = np.linalg.norm(joints(mean)[2] - target)
    ax.plot(*target, "*", ms=16, color=C["yellow"], mec=C.ink, mew=0.6, zorder=5)
    ax.text(target[0] + 0.08, target[1] - 0.17, "target $\\mathbf{p}$", fontsize=10)
    ax.text(0.98, 0.02, f"averaged arm misses the target by {miss:.2f} m\n(link lengths 1.0 m and 0.8 m)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=C.muted)
    ax.add_patch(Rectangle((-0.18, -0.12), 0.36, 0.12, fc=C.faint, ec=C.muted))
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(-0.6, 1.65)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig


@figure
def optimizers(C):
    a, b = 1.0, 25.0
    f = lambda p: 0.5 * (a * p[0] ** 2 + b * p[1] ** 2)  # noqa: E731
    grad = lambda p: np.array([a * p[0], b * p[1]])  # noqa: E731
    p0, n = np.array([-4.5, 1.2]), 40

    def run(kind):
        p, m, v, path = p0.copy(), np.zeros(2), np.zeros(2), [p0.copy()]
        for k in range(1, n + 1):
            g = grad(p)
            if kind == "sgd":
                p = p - 0.075 * g
            elif kind == "momentum":
                m = 0.8 * m + g
                p = p - 0.02 * m
            else:
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                p = p - 0.15 * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
            path.append(p.copy())
        return np.array(path)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.25))
    X, Y = np.meshgrid(np.linspace(-5, 1.5, 200), np.linspace(-1.6, 1.6, 200))
    ax.contour(X, Y, 0.5 * (a * X ** 2 + b * Y ** 2), levels=[0.5, 2, 5, 10, 20], colors=[C.faint], linewidths=0.8)
    runs = [("sgd", "gradient descent", "orange"), ("momentum", "momentum", "aqua"), ("adam", "Adam", "blue")]
    for kind, name, hue in runs:
        P = run(kind)
        ax.plot(P[:, 0], P[:, 1], "-o", color=C[hue], ms=2.5, lw=1.3, label=name)
        bx.semilogy([f(p) for p in P], color=C[hue], lw=1.6, label=name)
    ax.plot(0, 0, "*", color=C.ink, ms=10)
    ax.set_title("$\\mathcal{L}=\\frac{1}{2}(\\theta_1^2+25\\,\\theta_2^2)$: a narrow valley")
    ax.set_xlabel("$\\theta_1$ (flat direction)")
    ax.set_ylabel("$\\theta_2$ (steep)")
    ax.legend(loc="lower right", fontsize=8)
    bx.set_title("loss vs. iteration")
    bx.set_xlabel("iteration $k$")
    bx.grid(True, axis="y")
    return fig


@figure
def backprop_graph(C):
    fig, ax = canvas(8, 3.2)
    y = 1.6
    nodes = [("$X$", 0.45), ("$Z_1$", 2.45), ("$H_1$", 4.25), ("$\\hat Y$", 6.05), ("$\\mathcal{L}$", 7.55)]
    for t, x in nodes:
        box(ax, C, x - 0.3, y - 0.22, 0.6, 0.44, t, "blue", fs=12)
    ops = ["$\\cdot W_1+\\mathbf{b}_1$", "ReLU", "$\\cdot W_2+\\mathbf{b}_2$", "MSE"]
    back = ["$\\bar W_1=X^{\\top}\\bar Z_1$", "$\\bar Z_1=\\bar H_1\\odot[Z_1>0]$",
            "$\\bar H_1=\\bar{\\hat Y}\\,W_2^{\\top}$", "$\\bar{\\hat Y}=\\frac{2}{B}(\\hat Y-Y^\\ast)$"]
    for k in range(4):
        x0, x1 = nodes[k][1] + 0.32, nodes[k + 1][1] - 0.32
        arrow(ax, C, (x0, y + 0.1), (x1, y + 0.1), color=C["blue"])
        label(ax, C, (x0 + x1) / 2, y + 0.32, ops[k], fs=10, color=C["blue"])
        arrow(ax, C, (x1, y - 0.1), (x0, y - 0.1), color=C["orange"])
        label(ax, C, (x0 + x1) / 2, y - 0.4, back[k], fs=9.5, color=C["orange"])
    for x, w, g in [(1.45, "$W_1,\\mathbf{b}_1$", ""), (5.15, "$W_2,\\mathbf{b}_2$", "$\\bar W_2=H_1^{\\top}\\bar{\\hat Y}$")]:
        box(ax, C, x - 0.4, 2.55, 0.8, 0.38, w, "violet", fs=10)
        arrow(ax, C, (x, 2.55), (x, y + 0.48), color=C["violet"], lw=1)
        if g:
            label(ax, C, x + 0.95, 2.74, g, fs=9.5, color=C["orange"])
    label(ax, C, 0.1, 0.85, "forward: compute every intermediate tensor and store it", fs=9, color=C["blue"], ha="left")
    label(ax, C, 0.1, 0.55, "backward: each operation turns the gradient of its output into the gradient of its input"
          " (a vector–Jacobian product)", fs=9, color=C["orange"], ha="left")
    label(ax, C, 0.1, 0.25, "every gradient has the shape of the tensor it belongs to", fs=9, ha="left")
    return fig


def _clip(c, d, hw, hh):
    """Point where the segment from box centre c towards d leaves the box."""
    dx, dy = d[0] - c[0], d[1] - c[1]
    t = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return (c[0] + dx * t, c[1] + dy * t)


def gnode(ax, C, c, name, val, grad=None, form=None, hue="blue", hw=0.55, hh=0.45):
    """Graph node: formula (muted), forward value (blue), gradient (orange)."""
    box(ax, C, c[0] - hw, c[1] - hh, 2 * hw, 2 * hh, hue=hue, ls="-" if grad is not None else "--", lw=1.4)
    top = c[1] + hh - 0.17
    ax.text(c[0], top, form or name, ha="center", va="center", fontsize=8.5, color=C.muted)
    ax.text(c[0], c[1] - 0.02, f"{val:g}", ha="center", va="center", fontsize=12, color=C["blue"], fontweight="bold")
    if grad is not None:
        ax.text(c[0], c[1] - hh + 0.17, f"$\\bar{{{name}}}={grad:g}$", ha="center", va="center", fontsize=10, color=C["orange"])


def gedge(ax, C, a, b, tag=None, off=0.11, hw=0.55, hh=0.45, fwd=True):
    """Forward arrow a -> b (blue) and, beside it, the backward arrow b -> a (orange) labelled `tag`."""
    p, q = _clip(a, b, hw, hh), _clip(b, a, hw, hh)
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    n = (-(q[1] - p[1]) / L, (q[0] - p[0]) / L)
    sh = lambda pt, k: (pt[0] + n[0] * k, pt[1] + n[1] * k)
    if fwd:
        arrow(ax, C, sh(p, off), sh(q, off), color=C["blue"], lw=1.3)
    if tag is not None:
        arrow(ax, C, sh(q, -off), sh(p, -off), color=C["orange"], lw=1.3)
        m = sh(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), -0.3)
        ax.text(m[0], m[1], tag, ha="center", va="center", fontsize=9, color=C["orange"])


@figure
def backprop_scalar(C):
    x, w, b, v, y = 2.0, 1.5, -1.0, 2.0, 3.0

    def loss(x, w, b, v, y):
        return (v * max(w * x + b, 0.0) - y) ** 2

    p = w * x; z = p + b; h = max(z, 0.0); yh = v * h; L = loss(x, w, b, v, y)
    g_yh = 2 * (yh - y); g_h = v * g_yh; g_z = g_h * (z > 0); g_p = g_z
    g = dict(L=1.0, yh=g_yh, h=g_h, z=g_z, p=g_p, b=g_z, v=h * g_yh, w=x * g_p, x=w * g_p)
    for i, k in enumerate("xwbv"):  # the numbers drawn must match finite differences
        a = [x, w, b, v, y]; a[i] += 1e-6; hi = loss(*a); a[i] -= 2e-6
        assert abs((hi - loss(*a)) / 2e-6 - g[k]) < 1e-5, k

    fig, ax = canvas(9.0, 4.3)
    pos = dict(x=(0.75, 3.35), w=(0.75, 1.65), p=(2.4, 2.5), b=(3.9, 0.9), z=(3.9, 2.5), h=(5.4, 2.5),
               v=(6.9, 0.9), yh=(6.9, 2.5), y=(8.35, 0.9), L=(8.35, 2.5))
    forms = dict(p="$p=w\\,x$", z="$z=p+b$", h="$h=\\mathrm{ReLU}(z)$", yh="$\\hat y=v\\,h$", L="$\\mathcal{L}=(\\hat y-y)^2$")
    vals = dict(x=x, w=w, p=p, b=b, z=z, h=h, v=v, yh=yh, y=y, L=L)
    for k, c in pos.items():
        gnode(ax, C, c, "\\hat y" if k == "yh" else ("\\mathcal{L}" if k == "L" else k), vals[k],
              None if k == "y" else g[k], forms.get(k), hue="violet" if k in "wbv" else "blue")
    edges = [("x", "p", w), ("w", "p", x), ("p", "z", 1), ("b", "z", 1), ("z", "h", int(z > 0)), ("h", "yh", v),
             ("v", "yh", h), ("yh", "L", g_yh), ("y", "L", None)]
    for a, c, t in edges:
        gedge(ax, C, pos[a], pos[c], None if t is None else f"\u00d7{t:g}")
    label(ax, C, 0.1, 0.12, "blue: forward value     orange: $\\bar u=\\partial\\mathcal{L}/\\partial u$,"
          " each arrow multiplies by a local derivative     violet: parameters", fs=9, ha="left")
    return fig


@figure
def backprop_fanout(C):
    u = 2.0
    a, b = u * u, 3 * u
    ga, gb = 2 * u, 3.0
    fig, ax = canvas(6.6, 2.7)
    pos = dict(u=(0.8, 1.3), a=(3.2, 2.0), b=(3.2, 0.6), L=(5.7, 1.3))
    gnode(ax, C, pos["u"], "u", u, ga + gb, hw=0.5, hh=0.42)
    gnode(ax, C, pos["a"], "a", a, 1, "$a=u^2$", hw=0.5, hh=0.42)
    gnode(ax, C, pos["b"], "b", b, 1, "$b=3u$", hw=0.5, hh=0.42)
    gnode(ax, C, pos["L"], "\\mathcal{L}", a + b, 1, "$\\mathcal{L}=a+b$", hw=0.5, hh=0.42)
    kw = dict(hw=0.5, hh=0.42)
    gedge(ax, C, pos["u"], pos["a"], f"+{ga:g}", **kw)
    gedge(ax, C, pos["u"], pos["b"], f"+{gb:g}", **kw)
    gedge(ax, C, pos["a"], pos["L"], "\u00d71", **kw)
    gedge(ax, C, pos["b"], pos["L"], "\u00d71", **kw)
    label(ax, C, 0.8, 0.0, "$u$ is used twice, so its gradient\nis the sum of both paths: $4+3$", fs=9, color=C["orange"], va="center")
    return fig


@figure
def backprop_order(C):
    d = n = 1000
    rev, fwd = d * d + d * n, d * d * n + d * n
    fig, ax = canvas(8.6, 3.9)
    s = 0.85
    shapes = [("$J_3$", "1\u00d7d", 0.2), ("$J_2$", "d\u00d7d", s), ("$J_1$", "d\u00d7n", s)]
    xs = [0.5, 1.9, 3.3]
    for row, (hue, title, res, resh, cost, grp) in enumerate([
            ("orange", "reverse mode:  $(J_3 J_2)\\,J_1$", "1\u00d7d", 0.2, rev, (0.4, 2.85)),
            ("blue", "forward mode:  $J_3\\,(J_2 J_1)$", "d\u00d7n", s, fwd, (1.8, 4.25))]):
        yc = 2.75 - row * 1.85
        label(ax, C, 0.15, yc + 0.78, title, fs=10.5, color=C[hue], ha="left")
        for (nm, sh, h_), x0 in zip(shapes, xs):
            box(ax, C, x0, yc - h_ / 2, s, h_, hue="violet", r=0.03)
            label(ax, C, x0 + s / 2, yc + (0 if h_ > 0.5 else 0.2), nm, fs=11, color=C.ink)
            label(ax, C, x0 + s / 2, yc - 0.6, sh, fs=8.5)
        for x0 in (1.4, 2.8):
            label(ax, C, x0 + 0.25, yc, "\u00d7", fs=13, color=C.ink)
        box(ax, C, grp[0], yc - 0.52, grp[1] - grp[0], 1.04, hue=hue, fill=False, ls="--", r=0.1)
        arrow(ax, C, (4.35, yc), (5.0, yc), color=C[hue])
        label(ax, C, 4.7, yc + 0.22, "first", fs=8, color=C[hue])
        box(ax, C, 5.1, yc - resh / 2, s, resh, hue=hue, r=0.03)
        label(ax, C, 5.1 + s / 2, yc - 0.6, res, fs=8.5, color=C[hue])
        label(ax, C, 6.2, yc + 0.18, "intermediate\nresult", fs=8.5, ha="left", va="center")
        label(ax, C, 6.2, yc - 0.42, f"$\\approx 10^{{{round(math.log10(cost))}}}$ mult-adds", fs=10, color=C[hue], ha="left")
    label(ax, C, 0.15, 0.0, f"for $d=n={d}$: forward mode costs ~{fwd / rev:.0f}\u00d7 more \u2014 and the gap grows with $n$ (billions of parameters)", fs=9.5, ha="left")
    return fig


@figure
def backprop_layer(C):
    fig, ax = canvas(8.2, 3.5)
    box(ax, C, 2.8, 1.0, 2.6, 1.5, hue="blue", lw=1.6)
    label(ax, C, 4.1, 2.12, "layer  $\\mathbf{y}=f(\\mathbf{x};\\theta)$", fs=11, color=C.ink)
    label(ax, C, 4.1, 1.55, "keeps what its backward\nrule needs (e.g. $\\mathbf{x}$)", fs=8.5)
    box(ax, C, 3.6, 3.0, 1.0, 0.4, "$\\theta$", "violet", fs=11)
    # forward lane
    arrow(ax, C, (0.7, 2.1), (2.8, 2.1), color=C["blue"], lw=1.6)
    arrow(ax, C, (5.4, 2.1), (7.5, 2.1), color=C["blue"], lw=1.6)
    arrow(ax, C, (3.95, 3.0), (3.95, 2.5), color=C["violet"], lw=1.3)
    label(ax, C, 1.75, 2.35, "input $\\mathbf{x}$", fs=10, color=C["blue"])
    label(ax, C, 6.45, 2.35, "output $\\mathbf{y}$", fs=10, color=C["blue"])
    # backward lane
    arrow(ax, C, (7.5, 1.4), (5.4, 1.4), color=C["orange"], lw=1.6)
    arrow(ax, C, (2.8, 1.4), (0.7, 1.4), color=C["orange"], lw=1.6)
    arrow(ax, C, (4.25, 1.0), (4.25, 0.45), color=C["orange"], lw=1.3)
    label(ax, C, 6.45, 1.15, "$\\bar{\\mathbf{y}}$ arrives", fs=10, color=C["orange"])
    label(ax, C, 1.75, 1.15, "$\\bar{\\mathbf{x}}=\\bar{\\mathbf{y}}\\,\\partial\\mathbf{y}/\\partial\\mathbf{x}$", fs=10, color=C["orange"])
    label(ax, C, 4.25, 0.2, "$\\bar\\theta=\\bar{\\mathbf{y}}\\,\\partial\\mathbf{y}/\\partial\\theta$  (goes to the optimiser)", fs=10, color=C["orange"])
    label(ax, C, 0.1, 3.3, "A layer only needs its own local rule \u2014\nit never sees the rest of the network.", fs=9, ha="left", va="center")
    return fig


@figure
def backprop_affine(C):
    B, di, do, s = 4, 3, 2, 0.24
    fig, ax = canvas(8.6, 3.6)

    def mat(x, ytop, r, c, hue, shape, name=None):
        cells(ax, C, x, ytop, r, c, s, hue=hue)
        label(ax, C, x + c * s / 2, ytop - r * s - 0.14, shape, fs=8, color=C.muted)
        return x + c * s

    def prod(x0, ytop, title, hue, parts):
        label(ax, C, x0, ytop + 0.3, title, fs=10, color=C[hue], ha="left")
        x = x0
        for k, (r, c, hu, sh, sym) in enumerate(parts):
            x = mat(x, ytop, r, c, hu, sh)
            if sym:
                label(ax, C, x + 0.2, ytop - 0.5 * s * max(r, 2), "\u00d7" if sym == "\u00b7" else sym, fs=13, color=C.ink)
                x += 0.4

    prod(0.2, 3.1, "forward:  $Y=XW$", "blue", [(B, di, "blue", "B\u00d7d_in", "\u00b7"), (di, do, "violet", "d_in\u00d7d_out", "="), (B, do, "blue", "B\u00d7d_out", "")])
    prod(4.6, 3.1, "to the input:  $\\bar X=\\bar Y\\,W^{\\top}$", "orange", [(B, do, "orange", "B\u00d7d_out", "\u00b7"), (do, di, "violet", "d_out\u00d7d_in", "="), (B, di, "orange", "B\u00d7d_in", "")])
    prod(0.2, 1.4, "to the weights:  $\\bar W=X^{\\top}\\bar Y$  (sum over the batch)", "orange", [(di, B, "blue", "d_in\u00d7B", "\u00b7"), (B, do, "orange", "B\u00d7d_out", "="), (di, do, "orange", "d_in\u00d7d_out", "")])
    prod(4.6, 1.4, "to the bias:  $\\bar{\\mathbf{b}}=\\mathbf{1}^{\\top}\\bar Y$", "orange", [(1, B, "aqua", "1\u00d7B", "\u00b7"), (B, do, "orange", "B\u00d7d_out", "="), (1, do, "orange", "1\u00d7d_out", "")])
    return fig


@figure
def backprop_memory(C):
    L, k = 16, 4
    plain = [(1, i) for i in range(1, L + 1)] + [(2, L - i) for i in range(1, L + 1)]
    ck, kept = [], 0
    for i in range(1, L + 1):  # forward: keep every k-th activation
        kept += i % k == 0
        ck.append((1, kept + (i % k != 0)))
    for seg in range(L // k, 0, -1):
        kept -= 1  # this segment's checkpoint is no longer needed after it
        ck += [(1, kept + j) for j in range(1, k + 1)]  # recompute the segment
        ck += [(2, kept + k - j) for j in range(1, k + 1)]  # backward through it
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for steps, hue, lab in [(plain, "orange", "store everything"), (ck, "blue", f"checkpoint every {k} layers")]:
        t, m = [0], [0]
        for dur, mem in steps:
            t.append(t[-1] + dur); m.append(mem)
        ax.step(t, m, where="post", color=C[hue], label=f"{lab}: peak {max(m)}, time {t[-1]}")
    ax.axvline(L, color=C.faint, lw=1, ls=":")
    ax.text(L / 2, 17.5, "forward", ha="center", fontsize=8.5, color=C.muted)
    ax.text(L + 8, 17.5, "backward", ha="center", fontsize=8.5, color=C.muted)
    ax.set_ylim(0, 19)
    ax.set_xlabel("compute time  (forward layer = 1, backward layer = 2)")
    ax.set_ylabel("stored activations")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), fontsize=8)
    ax.grid(True, axis="y")
    return fig


@figure
def softmax_temperature(C):
    toks = ["station", "bay", "gate", "pallet", "drone"]
    z = np.array([2.0, 1.0, 0.5, -0.5, -1.5])
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, tau in zip(axs, [0.5, 1.0, 3.0]):
        p = np.exp(z / tau) / np.exp(z / tau).sum()
        H = -(p * np.log(p)).sum()
        ax.bar(range(5), p, color=C["blue"], width=0.62)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(range(5), toks, rotation=35, fontsize=8.5)
        ax.set_title(f"$\\tau={tau:g}$   entropy {H:.2f} nats")
        ax.set_ylim(0, 1.0)
    axs[0].set_ylabel("probability")
    axs[0].text(0.98, 0.95, "sharper", transform=axs[0].transAxes, ha="right", va="top", color=C.muted, fontsize=9)
    axs[2].text(0.98, 0.95, "flatter\n(max entropy log 5 = 1.61)", transform=axs[2].transAxes, ha="right", va="top",
                color=C.muted, fontsize=9)
    return fig


@figure
def mse_vs_categorical(C):
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-0.6, 0.07, 120), rng.normal(0.6, 0.07, 120)])
    xs = np.linspace(-1.2, 1.2, 400)
    mu, sd = y.mean(), y.std()
    bins = np.linspace(-1.2, 1.2, 25)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=False, gridspec_kw=dict(wspace=0.3))
    for ax in (a, b):
        ax.set_xlabel("grasp position along the toolbox (m)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    h, _, _ = a.hist(y, bins=bins, density=True, color=C.faint, label="demonstrations")
    a.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=C["orange"], label="Gaussian fit")
    a.axvline(mu, color=C["red"], ls="--", lw=1.5)
    a.text(mu + 0.04, h.max() * 1.0, "MSE prediction:\nthe lid, between\nthe handles", color=C["red"], fontsize=8.5, va="top")
    a.set_title("Gaussian head, trained with MSE")
    a.legend(loc="upper left", fontsize=8, ncol=2)
    counts, _ = np.histogram(y, bins=bins)
    p = counts / counts.sum()
    centres = (bins[1:] + bins[:-1]) / 2
    b.bar(centres, p, width=0.085, color=C["blue"])
    b.set_title("categorical head (24 bins), cross-entropy")
    for xpos, t in [(-0.6, "left handle"), (0.6, "right handle")]:
        b.text(xpos, p.max() * 1.04, t, ha="center", fontsize=8.5, color=C.ink)
    b.set_ylim(0, p.max() * 1.18)
    a.set_ylim(0, h.max() * 1.45)
    return fig


@figure
def gaussian_vs_categorical_head(C):
    # same grasp data as mse_vs_categorical, so the two figures agree
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-0.6, 0.07, 120), rng.normal(0.6, 0.07, 120)])
    p, _ = np.histogram(y, bins=np.linspace(-1.2, 1.2, 13))
    p = p / p.sum()
    fig, ax = canvas(8.1, 4.4)
    cols = dict(head=1.45, out=3.05, train=4.85, inf=6.6)
    for x, w, t in [(0.05, 1.0, "shared body"), (cols["head"], 1.25, "last layer"), (cols["out"], 1.5, "network output"),
                    (cols["train"], 1.5, "training"), (cols["inf"], 1.45, "inference")]:
        label(ax, C, x + w / 2, 4.1, t, fs=9.5, color=C.ink, fontweight="bold")
    box(ax, C, 0.05, 0.6, 1.0, 2.9, "network\nbody\n\n(encoder,\ntransformer,\nMLP)\n\nidentical\nin both", "violet", fs=8.5)
    rows = [(2.85, "orange", "Gaussian + MSE"), (1.05, "blue", "categorical + cross-entropy")]
    for yc, hue, name in rows:
        label(ax, C, 4.05, yc + 0.78, name, fs=8.5, color=C[hue], fontweight="bold")
        arrow(ax, C, (1.05, 2.05), (cols["head"], yc))
        arrow(ax, C, (cols["head"] + 1.25, yc), (cols["out"], yc))
        arrow(ax, C, (cols["out"] + 1.5, yc), (cols["train"], yc))
        arrow(ax, C, (cols["train"] + 1.5, yc), (cols["inf"], yc))
    # head
    box(ax, C, cols["head"], 2.5, 1.25, 0.7, "Linear(d, m)\nm numbers", "orange", fs=8.5, family="monospace")
    box(ax, C, cols["head"], 0.7, 1.25, 0.7, "Linear(d, m·V)\nV scores per dim", "blue", fs=8.5, family="monospace")
    # output
    x0, w = cols["out"], 1.5
    box(ax, C, x0, 2.4, w, 0.9, hue="orange")
    ax.plot([x0 + 0.15, x0 + w - 0.15], [2.65, 2.65], color=C.muted, lw=1)
    ax.plot([x0 + w / 2] * 2, [2.58, 2.72], color=C.muted, lw=1)
    ax.plot([x0 + w / 2], [2.65], "o", color=C["red"], ms=7)
    label(ax, C, x0 + w / 2, 3.0, "one number: 0.0", fs=8.5, color=C.ink)
    label(ax, C, x0 + w / 2, 2.5, "(the lid)", fs=8, color=C["red"])
    box(ax, C, x0, 0.6, w, 1.0, hue="blue")
    bw = (w - 0.3) / len(p)
    for i, v in enumerate(p):
        ax.add_patch(Rectangle((x0 + 0.15 + i * bw, 0.72), bw * 0.8, v / p.max() * 0.5, fc=C["blue"], ec="none"))
    label(ax, C, x0 + w / 2, 1.4, "V probabilities", fs=8.5, color=C.ink)
    # training
    box(ax, C, cols["train"], 2.4, 1.5, 0.9, "mse(pred, y)\ntarget y: a float", "orange", fs=8.5)
    box(ax, C, cols["train"], 0.6, 1.5, 1.0, "cross_entropy\ntarget: bin index k\n(discretise first)", "blue", fs=8.5)
    # inference
    box(ax, C, cols["inf"], 2.4, 1.45, 0.9, "action = output\ndeterministic", "orange", fs=8.5)
    box(ax, C, cols["inf"], 0.6, 1.45, 1.0, "sample k ~ softmax\naction = bin centre", "blue", fs=8.5)
    label(ax, C, 4.05, 0.2, "only the last layer, the loss and the decode step differ; the body is shared", fs=8.5)
    return fig


@figure
def head_nn_view(C):
    fig, ax = canvas(8.1, 6.6)
    X = dict(inp=0.7, h1=1.7, h2=2.7, out=4.9)

    def col(x, n, yc, gap):
        return x, yc + (np.arange(n)[::-1] - (n - 1) / 2) * gap

    def links(a, b, hue, alpha, lw=0.7):
        for ya in a[1]:
            for yb in b[1]:
                ax.plot([a[0], b[0]], [ya, yb], color=C[hue], lw=lw, alpha=alpha, zorder=1)

    def nodes(c, hue, fcs=None, r=0.1):
        for i, y in enumerate(c[1]):
            fc = fcs[i] if fcs is not None else C.soft(hue)
            ax.add_patch(Circle((c[0], y), r, fc=fc, ec=C[hue], lw=1.1, zorder=3))

    def body(yc):
        a, b, c = col(X["inp"], 3, yc, 0.4), col(X["h1"], 5, yc, 0.36), col(X["h2"], 5, yc, 0.36)
        links(a, b, "violet", 0.35)
        links(b, c, "violet", 0.35)
        for k in (a, b, c):
            nodes(k, "violet")
        return c

    # --- Gaussian: m output units, each one is the action itself
    yg = 5.0
    h = body(yg)
    o = col(X["out"], 2, yg, 0.7)
    links(h, o, "orange", 0.6, 0.9)
    nodes(o, "orange", r=0.13)
    for y, t in zip(o[1], ["$\\hat y_1$", "$\\hat y_2$"]):
        ax.text(X["out"], y, t, ha="center", va="center", fontsize=8, color=C.ink, zorder=4)
    label(ax, C, 5.35, yg, "$m=2$ outputs, each is the\naction value itself\n\nloss: MSE against the\nfloat target $y$", fs=8.5,
          color=C.ink, ha="left")
    label(ax, C, 0.1, 6.3, "Gaussian head", fs=10, color=C["orange"], fontweight="bold", ha="left")

    # --- categorical: V logits per action dimension, softmax inside each group
    yc = 1.95
    h = body(yc)
    probs = [np.array([0.46, 0.04, 0.0, 0.04, 0.46]), np.array([0.04, 0.2, 0.52, 0.2, 0.04])]
    centres = [yc + 0.85, yc - 0.85]
    label(ax, C, 0.1, 3.8, "categorical head", fs=10, color=C["blue"], fontweight="bold", ha="left")
    for g, (p, cy) in enumerate(zip(probs, centres)):
        o = col(X["out"], 5, cy, 0.26)
        links(h, o, "blue", 0.35, 0.6)
        nodes(o, "blue", [mix(C["blue"], C.surface, 0.15 + 0.85 * v / 0.52) for v in p], r=0.095)
        for y, v in zip(o[1], p):
            ax.text(X["out"] + 0.18, y, f"{v:.2f}", fontsize=7, va="center", color=C.muted)
        ax.plot([5.55, 5.6, 5.6, 5.55], [o[1][0] + 0.1, o[1][0] + 0.1, o[1][-1] - 0.1, o[1][-1] - 0.1], color=C.muted, lw=1)
        label(ax, C, 5.72, cy, f"softmax over\n$V=5$ bins of\naction dim {g + 1}", fs=8, color=C.ink, ha="left")
    label(ax, C, 5.72, yc, "loss: cross-entropy\nagainst the bin index", fs=8.5, color=C.ink, ha="left")
    # --- shared annotation
    ax.plot([0.1, 8.0], [3.95, 3.95], color=C.faint, lw=1)
    for x, t in [(X["inp"], "input"), (X["h1"], "hidden"), (X["h2"], "last hidden\nlayer $\\mathbf{h}$")]:
        label(ax, C, x, 0.3, t, fs=8.5)
    label(ax, C, 5.4, 0.3, "output layer: $m$ units  vs  $m\\cdot V$ units", fs=8.5, color=C.ink)
    return fig


@figure
def entropy_cross_kl(C):
    toks = ["left", "right", "up", "down"]
    p = np.array([0.50, 0.30, 0.15, 0.05])
    models = [("model = data", p, "green"), ("close model", np.array([0.40, 0.30, 0.20, 0.10]), "blue"),
              ("overconfident", np.array([0.05, 0.85, 0.05, 0.05]), "orange")]
    H = -(p * np.log(p)).sum()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.15]))
    w = 0.26
    a.bar(np.arange(4) - w, p, w, color=C.faint, label="data $p$")
    for k, (name, q, hue) in enumerate(models[1:]):
        a.bar(np.arange(4) + k * w, q, w, color=C[hue], label=name)
    a.set_xticks(range(4), toks)
    a.set_ylabel("probability")
    a.set_title("what the data does vs what the model says")
    a.legend(fontsize=7.5, loc="upper left")
    a.set_ylim(0, 1.05)
    ys = np.arange(3)[::-1]
    for y, (name, q, hue) in zip(ys, models):
        ce = -(p * np.log(q)).sum()
        b.barh(y, H, color=C.faint, height=0.55)
        b.barh(y, ce - H, left=H, color=C[hue], height=0.55)
        b.text(ce + 0.04, y, f"$H(p,q)$ = {ce:.2f}", va="center", fontsize=8.5)
        if ce - H > 0.02:
            b.text(H + (ce - H) / 2, y + 0.38, f"KL {ce - H:.2f}", ha="center", fontsize=8, color=C[hue])
    b.text(H / 2, ys[0] + 0.38, f"$H(p)$ = {H:.2f}", ha="center", fontsize=8, color=C.muted)
    b.set_yticks(ys, [m[0] for m in models], fontsize=8.5)
    b.set_xlim(0, 2.1)
    b.set_ylim(-0.5, 2.75)
    b.set_xlabel("nats per sample")
    b.set_title("cross-entropy = entropy + KL")
    return fig


@figure
def perplexity_example(C):
    toks = ["move", "the", "arm", "to", "the", "red", "cube", "."]
    p = np.array([0.30, 0.85, 0.40, 0.90, 0.80, 0.12, 0.55, 0.95])
    nll = -np.log(p)
    m = nll.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    a.bar(range(len(p)), nll, color=[C["red"] if v > 1.5 else C["blue"] for v in nll], width=0.62)
    for i, (v, pp) in enumerate(zip(nll, p)):
        a.text(i, v + 0.05, f"p={pp:.2f}", ha="center", fontsize=7.5)
    a.axhline(m, color=C["orange"], ls="--", lw=1.5)
    a.text(-0.4, 2.45, f"dashed: mean = {m:.2f} nats\nperplexity = $e^{{{m:.2f}}}$ = {np.exp(m):.2f}", ha="left",
           va="center", color=C["orange"], fontsize=8.5)
    a.set_xticks(range(len(p)), toks, fontsize=8.5)
    a.set_ylabel("surprise $-\\log p$ (nats)")
    a.set_ylim(0, nll.max() * 1.3)
    a.set_title("probability the model gave each true next token")
    x = np.linspace(0, 4.8, 200)
    b.semilogy(x, np.exp(x), color=C.muted, lw=1.5)
    pts = [("this sentence", m, "orange", -0.1, 1.0), ("weaker LM", 3.0, "blue", -0.1, 1.0),
           ("uniform over\n100 tokens", np.log(100), "red", -0.1, 1.0)]
    for name, v, hue, dx, _ in pts:
        b.plot([v], [np.exp(v)], "o", color=C[hue], ms=7)
        if name == "this sentence":
            b.text(v + 0.2, np.exp(v) * 0.8, f"{name}\nPPL {np.exp(v):.1f}", ha="left", va="center", fontsize=7.5, color=C[hue])
        else:
            b.text(v + dx, np.exp(v) * 1.25, f"{name}\nPPL {np.exp(v):.1f}", ha="right", va="bottom", fontsize=7.5, color=C[hue])
    b.set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    b.minorticks_off()
    b.set_xlabel("mean NLL per token (nats)")
    b.set_ylabel("perplexity $=e^{\\mathrm{NLL}}$")
    b.set_xlim(0, 4.9)
    b.set_ylim(0.9, 300)
    b.set_title("perplexity = effective number of choices", fontsize=9)
    print(f"perplexity example: mean NLL {m:.3f}, PPL {np.exp(m):.2f}, 'red' nll {nll[5]:.2f}")
    return fig


@figure
def bayes_rule(C):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.1), gridspec_kw=dict(wspace=0.15, width_ratios=[1, 1.3]))
    # (a) natural frequencies: 100 situations, 10 with an object; detector hits 90% / false-alarms 20%
    n_obj, hit, fa = 10, 9, 18
    for r in range(10):
        for c in range(10):
            present = r == 0
            detected = (c < hit) if present else (c < 2)
            a.scatter(c, -r - (0.5 if r else 0), s=95, color=C["orange"] if present else C.faint, zorder=2,
                      edgecolors=C.ink if detected else "none", linewidths=1.6)
    a.set_xlim(-0.7, 9.7)
    a.set_ylim(-9.8, 0.8)
    a.axis("off")
    a.set_title("100 cases; ringed = detector fired", fontsize=9)
    a.text(4.5, -11.2, f"{hit + fa} detections, {hit} real\n$p(\\mathrm{{object}}\\mid\\mathrm{{detection}})={hit}/{hit + fa}={hit / (hit + fa):.2f}$",
           ha="center", va="top", fontsize=8.5, clip_on=False)
    a.text(4.5, -10.0, "orange: object (10)   grey: no object (90)", color=C.muted, fontsize=8, ha="center", va="top")
    # (b) Gaussian prior x likelihood -> narrower posterior (the Kalman-filter update in 1-D)
    m0, s0, m1, s1 = 0.0, 0.6, 1.0, 0.4
    w0, w1 = 1 / s0**2, 1 / s1**2
    mp, sp = (w0 * m0 + w1 * m1) / (w0 + w1), (w0 + w1) ** -0.5
    x = np.linspace(-2, 3, 400)
    g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    for m, s, hue, name in [(m0, s0, "blue", "prior\n(where we expected)"), (m1, s1, "orange", "likelihood\n(sensor reading)"),
                            (mp, sp, "green", "posterior")]:
        b.plot(x, g(m, s), color=C[hue], lw=2)
        b.fill_between(x, g(m, s), color=C[hue], alpha=0.12 if hue != "green" else 0.25)
    b.text(-0.55, g(m0, s0).max() * 0.5, "prior", color=C["blue"], ha="right", fontsize=8.5)
    b.text(1.55, g(m1, s1).max() * 0.55, "likelihood", color=C["orange"], ha="left", fontsize=8.5)
    b.text(mp, g(mp, sp).max() * 1.04, f"posterior  $\\mu={mp:.2f},\\ \\sigma={sp:.2f}$", color=C["green"], ha="center", fontsize=8.5)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.set_xlabel("object position (m)")
    b.set_ylim(0, g(mp, sp).max() * 1.2)
    b.set_title("continuous: prior × likelihood, renormalised", fontsize=9)
    print(f"bayes: post p={hit / (hit + fa):.3f}; gaussian mu={mp:.3f} sigma={sp:.3f} (prior sd {s0}, lik sd {s1})")
    return fig


@figure
def kl_forward_reverse(C):
    x = np.linspace(-6, 6, 1201)
    dx = x[1] - x[0]
    gauss = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
    p = 0.5 * gauss(-2.6, 0.6) + 0.5 * gauss(2.6, 0.7)
    mf, sf = (x * p).sum() * dx, np.sqrt(((x - (x * p).sum() * dx) ** 2 * p).sum() * dx)  # forward KL = moment match
    best = min(((((q := gauss(m, s)) * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum() * dx, m, s)
                for m in np.linspace(-3, 3, 121) for s in np.linspace(0.3, 3, 55)), key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.fill_between(x, p, color=C.faint, label="data $p$ (two modes)")
    ax.plot(x, gauss(mf, sf), color=C["orange"], label="argmin $D_{KL}(p\\,\\|\\,q)$: forward, mass-covering")
    ax.plot(x, gauss(best[1], best[2]), color=C["blue"], label="argmin $D_{KL}(q\\,\\|\\,p)$: reverse, mode-seeking")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, 0.95)
    ax.set_title("fitting one Gaussian $q$ to a bimodal $p$", fontsize=9.5)
    return fig


def bpe_steps(word_counts, n_merges):
    seqs = {w: list(w) for w in word_counts}
    history = [({w: list(s) for w, s in seqs.items()}, None, 0)]
    for _ in range(n_merges):
        pairs = {}
        for w, f in word_counts.items():
            s = seqs[w]
            for pair in zip(s, s[1:]):
                pairs[pair] = pairs.get(pair, 0) + f
        (a, b), cnt = max(pairs.items(), key=lambda kv: kv[1])
        for w, s in seqs.items():
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
        history.append(({w: list(s) for w, s in seqs.items()}, (a, b), cnt))
    return history


@figure
def bpe_merges(C):
    counts = {"drone": 5, "drones": 3, "zone": 2, "crate": 4, "crates": 2}
    hist = bpe_steps(counts, 6)
    shown = ["drone", "drones", "zone", "crates"]
    fig, ax = canvas(7.8, 3.75)
    xs = [1.95, 3.35, 4.95, 6.2]
    for w, x in zip(shown, xs):
        label(ax, C, x + 0.45, 3.55, f"{w} ×{counts[w]}", fs=9.5, color=C.ink)
    for r, (seqs, merge, cnt) in enumerate(hist):
        y = 3.1 - r * 0.45
        txt = "start: characters" if merge is None else f"{r}. merge {merge[0]}+{merge[1]} ({cnt})"
        label(ax, C, 0.05, y + 0.14, txt, fs=8.5, ha="left", color=C.ink)
        for w, x in zip(shown, xs):
            new = merge[0] + merge[1] if merge else None
            hues = ["orange" if s == new else "blue" for s in seqs[w]]
            chips(ax, C, x, y, seqs[w], hues, h=0.28, fs=9, gap=0.03, mono=True)
    return fig


@figure
def tokenization_granularity(C):
    text = "Survey the east yard, find the red toolbox, and bring it to the inspection station."
    bpe = ["Sur", "vey", "␣the", "␣east", "␣yard", ",", "␣find", "␣the", "␣red", "␣toolbox", ",", "␣and", "␣bring",
           "␣it", "␣to", "␣the", "␣inspection", "␣station", "."]
    fig, ax = canvas(8, 3.2)

    def rows(units, y, fs, maxw=7.9):
        x, alt = 0.05, 0
        for u in units:
            w = chip_w(u, fs)
            if x + w > maxw:
                x, y = 0.05, y - 0.34
            box(ax, C, x, y, w, 0.27, u, ["blue", "aqua"][alt], fs=fs, r=0.04, family="monospace")
            x += w + 0.03
            alt ^= 1
        return y

    label(ax, C, 0.05, 3.05, f"characters: T = {len(text)} (each byte or character is a token)", fs=9.5, ha="left", color=C.ink)
    y = rows([c if c != " " else "␣" for c in text], 2.6, 6.5, maxw=7.95)
    label(ax, C, 0.05, y - 0.2, f"BPE subwords (SmolLM2, V = 49,152): T = {len(bpe)}", fs=9.5, ha="left", color=C.ink)
    y = rows(bpe, y - 0.68, 8.5)
    label(ax, C, 0.05, y - 0.2, f"words: T = {len(text.split())}, but every unseen word is out of vocabulary",
          fs=9.5, ha="left", color=C.ink)
    rows(text.split(), y - 0.68, 8.5)
    return fig


@figure
def embedding_lookup(C):
    fig, ax = canvas(7.6, 3.15)
    V, d, s, i = 8, 5, 0.27, 3
    top = 2.7
    oh = [["1" if c == i else "0" for c in range(V)]]
    fills = [[C.soft("orange", 0.6) if c == i else C.soft("blue", 0.08) for c in range(V)]]
    cells(ax, C, 0.2, top - 3 * s, 1, V, s, "blue", fills=fills, texts=oh, fs=8, tc=C.ink)
    label(ax, C, 0.2 + V * s / 2, top - 3 * s + 0.2, "one-hot $\\mathbf{e}_i$ (1 × V)", fs=9.5, color=C.ink)
    label(ax, C, 0.2 + (i + .5) * s, top - 4 * s - 0.15, "id 3\n'toolbox'", fs=8, color=C["orange"], va="top")
    label(ax, C, 2.65, top - 3.5 * s, "×", fs=16, color=C.ink)
    rng = np.random.default_rng(1)
    E = rng.normal(size=(V, d))
    cmap, norm = C.div(), plt.Normalize(-2.2, 2.2)
    fl = [[cmap(norm(v)) for v in r] for r in E]
    cells(ax, C, 3.0, top, V, d, s, "blue", fills=fl, edge=C.surface)
    ax.add_patch(Rectangle((3.0, top - (i + 1) * s), d * s, s, fc="none", ec=C["orange"], lw=2.2))
    label(ax, C, 3.0 + d * s / 2, top + 0.2, "embedding table $E$ (V × d)", fs=9.5, color=C.ink)
    label(ax, C, 3.05 + d * s, top - (i + .5) * s, "row 3", fs=8, color=C["orange"], ha="left")
    label(ax, C, 5.05, top - 3.5 * s, "=", fs=16, color=C.ink)
    cells(ax, C, 5.4, top - 3 * s, 1, d, s, "orange", fills=[fl[i]], edge=C["orange"], lw=1.6)
    label(ax, C, 5.4 + d * s / 2, top - 3 * s + 0.2, "$\\mathbf{x}=E_{3,:}$ (1 × d)", fs=9.5, color=C.ink)
    label(ax, C, 3.8, 0.12, "multiplying by a one-hot row selects a row; code does it by indexing: E[ids]", fs=9)
    return fig


@figure
def embedding_space(C):
    groups = {  # hand-placed so that labels do not collide
        "vehicles": ("blue", (1.0, 1.25), {"drone": (1.0, 0.55), "quadrotor": (0.7, 0.85), "UAV": (1.35, 0.85),
                                            "rover": (1.3, 0.3)}),
        "objects": ("orange", (-1.0, 1.25), {"toolbox": (-0.9, 0.5), "crate": (-1.35, 0.3), "pallet": (-0.7, 0.85),
                                              "box": (-1.35, 0.8)}),
        "colours": ("aqua", (-0.95, -1.5), {"red": (-0.8, -0.75), "blue": (-0.4, -1.0), "yellow": (-1.35, -0.95),
                                             "green": (-0.85, -1.2)}),
        "places": ("violet", (1.0, -1.5), {"station": (0.65, -0.8), "yard": (1.25, -0.6), "bay": (1.25, -1.05),
                                            "gate": (0.75, -1.2)}),
    }
    pts = {}
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for g, (hue, at, words) in groups.items():
        for w, xy in words.items():
            pts[w] = np.array(xy)
            ax.plot(*pts[w], "o", color=C[hue], ms=7)
            ax.text(pts[w][0] + 0.07, pts[w][1], w, fontsize=9, color=C.ink, va="center")
        ax.text(*at, g, ha="center", va="center", fontsize=10, color=C[hue], weight="bold")
    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)  # noqa: E731
    for w, hue in [("drone", "blue"), ("quadrotor", "blue"), ("toolbox", "orange")]:
        ax.annotate("", xy=pts[w], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C[hue], lw=1.2, alpha=0.8))
    ax.plot(0, 0, "o", color=C.muted, ms=4)
    ax.text(0, -0.12, f"cos(drone, quadrotor) = {cos(pts['drone'], pts['quadrotor']):.2f}\n"
                         f"cos(drone, toolbox) = {cos(pts['drone'], pts['toolbox']):.2f}",
            fontsize=9, color=C.ink, ha="center", va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title("illustrative 2-D slice of an embedding space", fontsize=9.5, color=C.muted)
    return fig


@figure
def seq2seq_attention(C):
    fig, ax = canvas(8, 3.2)
    src = ["find", "the", "red", "toolbox"]

    def encoder(x0):
        xs = [x0 + k * 0.82 for k in range(4)]
        for k, (x, t) in enumerate(zip(xs, src)):
            box(ax, C, x - 0.3, 1.0, 0.6, 0.38, f"$\\mathbf{{s}}_{k + 1}$", "blue", fs=11)
            label(ax, C, x, 0.72, t, fs=9, color=C.ink)
            if k:
                arrow(ax, C, (x - 0.52, 1.19), (x - 0.3, 1.19), color=C["blue"], lw=1)
        return xs

    xs = encoder(0.55)
    box(ax, C, 1.08, 2.25, 1.4, 0.42, "one vector $\\mathbf{h}$", "red", fs=10)
    for x in xs:
        arrow(ax, C, (x, 1.38), (1.78, 2.25), color=C.faint if x != xs[-1] else C["red"], lw=1 if x != xs[-1] else 1.6)
    label(ax, C, 1.78, 3.0, "(a) seq2seq: everything passes\nthrough one fixed-size state", fs=9.5, color=C.ink)
    label(ax, C, 3.15, 2.05, "decoder\nstarts here", fs=8, ha="left")

    xs = encoder(4.65)
    alpha = np.array([0.08, 0.05, 0.27, 0.60])
    dec = (5.88, 2.45)
    box(ax, C, dec[0] - 0.45, dec[1], 0.9, 0.42, "$\\mathbf{h}_{t-1}$", "violet", fs=11)
    for x, a in zip(xs, alpha):
        ax.plot([x, dec[0]], [1.38, dec[1]], color=C["violet"], lw=0.6 + 7 * a, alpha=0.3 + 0.7 * a, solid_capstyle="round")
        label(ax, C, x + (0.18 if x > dec[0] else -0.18), 1.62, f"{a:.2f}", fs=8, color=C["violet"])
    label(ax, C, 5.88, 3.12, "(b) attention: the decoder reads every state,\nweighted by relevance $\\alpha_{t,j}$",
          fs=9.5, color=C.ink)
    label(ax, C, 5.88, 0.3, "$\\mathbf{c}_t=\\sum_j \\alpha_{t,j}\\,\\mathbf{s}_j$", fs=11, color=C.ink)
    return fig


@figure
def vanishing_gradients(C):
    d, T = 64, 60
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for rho, hue in [(0.8, "blue"), (1.0, "aqua"), (1.6, "orange")]:
        norms = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            W = rng.normal(size=(d, d)) / math.sqrt(d) * rho
            Wx = rng.normal(size=(d, d)) / math.sqrt(d)
            h, J, ns = np.zeros(d), np.eye(d), [1.0]
            for _ in range(T):
                h = np.tanh(h @ W + rng.normal(size=d) @ Wx * 0.5)
                J = J @ W @ np.diag(1 - h ** 2)  # row-vector convention: dh_t/dh_0 accumulates on the right
                ns.append(np.linalg.norm(J, 2))
            norms.append(ns)
        m = np.exp(np.log(np.array(norms) + 1e-30).mean(0))
        ax.semilogy(m, color=C[hue], label=f"$W_h$ scaled to gain {rho:g}")
    ax.set_xlabel("steps back in time  $t-k$")
    ax.set_ylabel("$\\|\\partial\\mathbf{h}_t/\\partial\\mathbf{h}_k\\|_2$")
    ax.set_title("tanh RNN, d = 64: gradients from distant steps shrink geometrically", fontsize=9.5)
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y")
    return fig


def depot_scene(n=128):
    """A cartoon top-down depot image, float RGB in [0, 1]."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.empty((n, n, 3))
    img[:] = (0.72, 0.70, 0.66)
    img[(np.abs(xx - 0.18) < 0.025)] = (0.93, 0.78, 0.15)                       # yellow lane line
    img[(xx > 0.55) & (xx < 0.9) & (yy > 0.08) & (yy < 0.36)] = (0.29, 0.45, 0.75)  # blue crate
    img[(xx > 0.58) & (xx < 0.87) & (yy > 0.2) & (yy < 0.23)] = (0.2, 0.32, 0.58)
    img[(xx > 0.35) & (xx < 0.62) & (yy > 0.55) & (yy < 0.77)] = (0.83, 0.2, 0.17)  # red toolbox
    img[(xx > 0.44) & (xx < 0.53) & (yy > 0.6) & (yy < 0.64)] = (0.15, 0.15, 0.15)  # handle
    img[(xx > 0.7) & (xx < 0.95) & (yy > 0.62) & (yy < 0.9)] = (0.55, 0.42, 0.25)  # pallet
    img[(xx > 0.7) & (xx < 0.95) & (((yy * n).astype(int) % 7) < 2) & (yy > 0.62) & (yy < 0.9)] = (0.42, 0.31, 0.17)
    return img


@figure
def vit_patches(C):
    img = depot_scene(128)
    g, P = 4, 32
    fig, ax = canvas(8, 3.35)
    x0, y0, side = 0.15, 0.6, 2.1
    ax.imshow(img, extent=(x0, x0 + side, y0, y0 + side), interpolation="bilinear")
    st = side / g
    for k in range(g + 1):
        ax.plot([x0 + k * st] * 2, [y0, y0 + side], color="white", lw=1.5)
        ax.plot([x0, x0 + side], [y0 + k * st] * 2, color="white", lw=1.5)
    for r in range(g):
        for c in range(g):
            ax.text(x0 + c * st + 0.05, y0 + side - r * st - 0.06, str(r * g + c + 1), fontsize=7, color="white",
                    va="top", weight="bold")
    label(ax, C, x0 + side / 2, 0.3, "image 3 × 128 × 128, P = 32\n→ 4 × 4 = 16 patches", fs=8.5, color=C.ink)
    arrow(ax, C, (2.4, 1.6), (2.75, 1.6))
    # patch sequence
    ps, gap, sx = 0.27, 0.04, 2.85
    for k in range(16):
        r, c = divmod(k, g)
        patch = img[r * P:(r + 1) * P, c * P:(c + 1) * P]
        xk = sx + k * (ps + gap)
        ax.imshow(patch, extent=(xk, xk + ps, 1.75, 1.75 + ps), interpolation="bilinear")
        label(ax, C, xk + ps / 2, 2.15, str(k + 1), fs=7, color=C.ink)
        arrow(ax, C, (xk + ps / 2, 1.73), (xk + ps / 2, 1.4), lw=0.8, ms=7)
        box(ax, C, xk, 0.7, ps, 0.6, "", "violet", r=0.03, lw=0.9)
    label(ax, C, sx + 8 * (ps + gap), 3.2, "a sequence of 16 patches, flattened row by row (raster order)", fs=9, color=C.ink)
    label(ax, C, sx + 8 * (ps + gap), 1.53, "flatten (3P² = 3072 numbers) → linear map $W_E$ → + position $\\mathbf{p}_k$",
          fs=8.5, color=C["violet"], bbox=dict(fc=C.surface, ec="none", pad=1))
    label(ax, C, sx + 8 * (ps + gap), 0.35, "16 image tokens of width d, read by a transformer encoder", fs=9, color=C.ink)
    # vertical neighbours are far apart in the sequence
    xa, xb = sx + 0 * (ps + gap) + ps / 2, sx + 4 * (ps + gap) + ps / 2
    arrow(ax, C, (xa, 2.3), (xb, 2.3), color=C["orange"], rad=-0.35, lw=1, head=False)
    label(ax, C, xb + 0.25, 2.55, "patches 1 and 5 touch in the image, but are 4 apart in the sequence",
          fs=8, color=C["orange"], ha="left")
    return fig


# ======================================================================= Chapter 1

@figure
def decoding_strategies(C):
    toks = ["inspection", "station", "gate", "east", "charging", "rover", "loading", "pallet", "yard", "drone"]
    p = np.array([0.42, 0.18, 0.11, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    cum = np.cumsum(p)
    nucleus = (cum - p) < 0.8
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(range(10), p, color=[C["blue"] if n else C.faint for n in nucleus], width=0.65)
    ax.bar(0, p[0], width=0.65, fc="none", ec=C["orange"], lw=2.2)
    ax.text(0.42, 0.3, "← greedy", ha="left", color=C["orange"], fontsize=9)
    ax.plot(range(10), cum, "-o", color=C.muted, lw=1.2, ms=3.5)
    ax.text(9, cum[-1] - 0.06, "cumulative", ha="right", color=C.muted, fontsize=8.5, va="top")
    for (x0, x1, y, t) in [(0, 2, 0.9, "top-k, k = 3"), (0, nucleus.sum() - 1, 1.02, "nucleus, top-p = 0.8")]:
        ax.plot([x0 - 0.3, x0 - 0.3, x1 + 0.3, x1 + 0.3], [y - 0.03, y, y, y - 0.03], color=C.ink, lw=1)
        ax.text((x0 + x1) / 2, y + 0.015, t, ha="center", fontsize=8.5)
    ax.set_xticks(range(10), ["␣" + t for t in toks], rotation=35, fontsize=8.5, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("$p(x_t\\mid$ prefix$)$")
    ax.set_title("next token after \"Bring the red toolbox to the\" (illustrative probabilities)", fontsize=9.5)
    return fig


@figure
def pixel_distance(C):
    n, pad = 96, 6
    rng = np.random.default_rng(7)
    N = rng.normal(size=(n + 2 * pad, n + 2 * pad))
    N = (N + np.roll(N, 1, 0) + np.roll(N, 1, 1) + np.roll(N, (1, 1), (0, 1))) / 2
    world = np.stack([0.62 + 0.09 * N, 0.6 + 0.09 * N, 0.56 + 0.09 * N], -1)

    def put(w, obj):
        w = w.copy()
        r0, c0 = pad + 38, pad + 28
        w[r0:r0 + 26, c0:c0 + 40] = (0.82, 0.18, 0.15)
        if obj == "toolbox":
            w[r0 - 8:r0 - 5, c0 + 12:c0 + 28] = 0.12
            w[r0 - 5:r0, c0 + 12:c0 + 15] = 0.12
            w[r0 - 5:r0, c0 + 25:c0 + 28] = 0.12
            w[r0 + 11:r0 + 13, c0:c0 + 40] = (0.55, 0.1, 0.08)
        else:
            for k in range(3, 26, 6):
                w[r0 + k:r0 + k + 2, c0:c0 + 40] = (0.6, 0.12, 0.1)
        return np.clip(w, 0, 1)

    view = lambda w, s: w[pad:pad + n, pad + s:pad + s + n]  # noqa: E731
    A = view(put(world, "toolbox"), 0)
    B = view(put(world, "toolbox"), 3)
    Cc = view(put(world, "crate"), 0)
    dAB, dAC = np.linalg.norm(A - B), np.linalg.norm(A - Cc)
    fig, ax = canvas(7.6, 3.2)
    for x, im, t in [(0.3, A, "A: red toolbox"), (2.9, B, "B: same scene,\ncamera drifted 3 px"),
                     (5.5, Cc, "C: a red crate\ninstead")]:
        ax.imshow(im, extent=(x, x + 1.8, 1.25, 3.05))
        label(ax, C, x + 0.9, 0.95, t, fs=9, color=C.ink)
    label(ax, C, 3.8, 0.42, f"pixel distance  ‖A − B‖ = {dAB:.1f}   (same object, 3-pixel shift)", fs=9.5, color=C.ink)
    label(ax, C, 3.8, 0.15, f"pixel distance  ‖A − C‖ = {dAC:.1f}   (different object, judged closer)", fs=9.5,
          color=C.ink)
    return fig


@figure
def fusion_mechanisms(C):
    fig, ax = canvas(8, 5.6)

    def panel(x, y, title, cost):
        box(ax, C, x, y, 3.85, 2.55, "", None, fill=False, r=0.08, lw=0.8, ls=(0, (3, 2)))
        label(ax, C, x + 0.12, y + 2.36, title, fs=10, color=C.ink, ha="left", weight="bold")
        label(ax, C, x + 3.73, y + 0.14, cost, fs=8.5, ha="right")

    # 1. concatenation
    panel(0.05, 2.95, "1  concatenate in the sequence", "cost $O((N_p+T)^2)$")
    xs = chips(ax, C, 0.3, 3.9, ["img"] * 4 + ["red", "box"], ["blue"] * 4 + ["orange"] * 2, h=0.28, fs=8, gap=0.06)
    for i, a in enumerate(xs):
        for b in xs[i + 1:]:
            ax.add_patch(Arc(((a + b) / 2, 4.18), b - a, (b - a) * 0.55, theta1=0, theta2=180, color=C.muted, lw=0.5))
    box(ax, C, 0.3, 3.45, 3.3, 0.28, "self-attention: every token reads every token", "violet", fs=8)
    # 2. cross-attention
    panel(4.1, 2.95, "2  cross-attention", "cost $O(TS)$")
    q = chips(ax, C, 4.75, 4.6, ["$h_1$", "$h_2$", "$h_3$"], "violet", h=0.28, fs=9, widths=0.42, gap=0.25)
    kv = chips(ax, C, 4.35, 3.35, ["$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$"], "blue", h=0.28, fs=9,
               widths=0.36, gap=0.14)
    for a in q:
        for b in kv:
            ax.plot([b, a], [3.63, 4.6], color=C["violet"], lw=0.6, alpha=0.6)
    label(ax, C, 4.3, 4.05, "queries\nfrom H", fs=8, ha="left", color=C["violet"])
    label(ax, C, 4.3, 3.13, "keys, values from C (C is not updated)", fs=8, ha="left", color=C["blue"])
    # 3. FiLM
    panel(0.05, 0.25, "3  FiLM", "cost $O(d)$ per token")
    box(ax, C, 0.25, 1.75, 0.55, 0.32, "$\\mathbf{c}$", "orange", fs=10)
    arrow(ax, C, (0.8, 1.91), (1.05, 1.91))
    box(ax, C, 1.05, 1.75, 0.55, 0.32, "MLP", "orange", fs=8.5)
    arrow(ax, C, (1.6, 1.98), (1.95, 2.12))
    arrow(ax, C, (1.6, 1.84), (1.95, 1.7))
    gam = [1.6, 0.3, 1.0, 0.6, 1.3]
    cells(ax, C, 2.0, 2.27, 1, 5, 0.2, "orange", fills=[[C.soft("orange", min(1, 0.15 + 0.5 * g)) for g in gam]])
    cells(ax, C, 2.0, 1.85, 1, 5, 0.2, "orange", fills=[[C.soft("orange", 0.2)] * 5])
    label(ax, C, 3.15, 2.17, "$\\gamma(\\mathbf{c})$", fs=10, ha="left", color=C.ink)
    label(ax, C, 3.15, 1.75, "$\\beta(\\mathbf{c})$", fs=10, ha="left", color=C.ink)
    fl = [[C.soft("blue", min(1, 0.12 + 0.45 * g)) for g in gam] for _ in range(4)]
    cells(ax, C, 2.0, 1.35, 4, 5, 0.2, "blue", fills=fl)
    label(ax, C, 1.85, 0.95, "H (T × d):\nevery token\nscaled the\nsame way", fs=7.5, ha="right", color=C.ink)
    label(ax, C, 3.15, 0.95, "$\\gamma\\odot\\mathbf{h}+\\beta$", fs=10, ha="left", color=C.ink)
    # 4. adaLN
    panel(4.1, 0.25, "4  adaptive LayerNorm (adaLN)", "cost $O(d)$ per token")
    ys = 1.45
    box(ax, C, 4.3, ys, 0.45, 0.34, "$\\mathbf{h}$", "blue", fs=10)
    arrow(ax, C, (4.75, ys + 0.17), (4.95, ys + 0.17))
    box(ax, C, 4.95, ys, 1.0, 0.34, "LayerNorm", "blue", fs=8.5)
    arrow(ax, C, (5.95, ys + 0.17), (6.15, ys + 0.17))
    box(ax, C, 6.15, ys, 0.95, 0.34, "$\\gamma(\\tau)\\odot\\,\\cdot+\\beta(\\tau)$", "orange", fs=8.5)
    arrow(ax, C, (7.1, ys + 0.17), (7.3, ys + 0.17))
    label(ax, C, 7.5, ys + 0.17, "out", fs=9, color=C.ink)
    box(ax, C, 6.15, 0.62, 0.95, 0.32, "MLP", "orange", fs=8.5)
    arrow(ax, C, (6.62, 0.94), (6.62, ys))
    box(ax, C, 5.0, 0.62, 0.85, 0.32, "time $\\tau$", "orange", fs=8.5)
    arrow(ax, C, (5.85, 0.78), (6.15, 0.78))
    label(ax, C, 4.3, 2.05, "the condition (e.g. diffusion time)\nreplaces the norm's fixed $\\gamma, \\beta$", fs=8,
          ha="left", color=C.ink)
    return fig


# ======================================================================= Chapter 2

@figure
def attention_steps(C):
    import torch

    torch.manual_seed(0)
    T, d, d_k = 4, 8, 4
    X = torch.randn(T, d)
    W_Q, W_K, W_V = (torch.randn(d, d_k) / math.sqrt(d) for _ in range(3))
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / math.sqrt(d_k)
    A = S.softmax(-1)
    Y = A @ V
    toks = ["Survey", "the", "east", "yard"]
    mats = [("$X$  (T×d)", X, "div", None), ("$Q=XW_Q$", Q, "div", None), ("$K=XW_K$", K, "div", None),
            ("$V=XW_V$", V, "div", None), ("$S=QK^{\\top}/\\sqrt{d_k}$", S, "div", "{:.1f}"),
            ("$A=\\mathrm{softmax}(S)$", A, "seq", "{:.2f}"), ("$Y=AV$", Y, "div", None)]
    fig = plt.figure(figsize=(8, 4.6))
    top = fig.add_gridspec(1, 4, width_ratios=[8, 4, 4, 4], wspace=0.3, left=0.1, right=0.98, top=0.93, bottom=0.66)
    bot = fig.add_gridspec(1, 3, wspace=0.45, left=0.1, right=0.98, top=0.5, bottom=0.08)
    slots = [top[0], top[1], top[2], top[3], bot[0], bot[1], bot[2]]
    for k, ((t, M, cm, ann), slot) in enumerate(zip(mats, slots)):
        ax = fig.add_subplot(slot)
        M = M.detach().numpy()
        cmap = C.div() if cm == "div" else C.seq()
        lim = (-2.2, 2.2) if cm == "div" else (0, 1)
        cols = toks if k in (4, 5) else None
        heat(ax, C, M, cmap, *lim, rows=toks if k in (0, 4, 5, 6) else None, cols=cols, annot=ann, fs=8)
        ax.set_title(t, fontsize=10)
        if cols:
            ax.set_xlabel("key $j$", fontsize=8.5, color=C.muted)
    fig.text(0.54, -0.1, "rows are queries, one per token.  Each row of $A$ sums to 1; "
             "row $i$ of $Y$ is $\\sum_j A_{ij}\\,\\mathbf{v}_j$.", ha="center", fontsize=9, color=C.muted)
    return fig


@figure
def soft_dictionary(C):
    fig, ax = canvas(8, 3.3)
    keys = ["Survey", "the", "east", "yard"]
    hues = ["blue", "aqua", "orange", "violet"]
    # hard lookup
    label(ax, C, 1.65, 3.1, "dictionary: exact match, one value", fs=10, color=C.ink, weight="bold")
    box(ax, C, 0.1, 1.75, 0.95, 0.38, "query 'east'", "red", fs=8.5)
    for k, (t, h) in enumerate(zip(keys, hues)):
        y = 2.5 - k * 0.5
        box(ax, C, 1.55, y, 0.8, 0.34, f"key {t}", h, fs=8)
        box(ax, C, 2.55, y, 0.6, 0.34, "$\\mathbf{v}$", h, fs=10, fill=k == 2)
        hit = k == 2
        arrow(ax, C, (1.05, 1.94), (1.55, y + 0.17), color=C["red"] if hit else C.faint, lw=1.6 if hit else 0.8,
              head=hit, ls="-" if hit else (0, (2, 2)))
    label(ax, C, 1.65, 0.35, "not differentiable: a match is all or nothing", fs=8.5)
    # soft lookup
    w = np.array([0.08, 0.04, 0.63, 0.25])
    label(ax, C, 5.75, 3.1, "attention: similarity-weighted blend of all values", fs=10, color=C.ink, weight="bold")
    box(ax, C, 3.95, 1.75, 0.85, 0.38, "$\\mathbf{q}$ ('yard')", "red", fs=9)
    for k, (t, h) in enumerate(zip(keys, hues)):
        y = 2.5 - k * 0.5
        box(ax, C, 5.2, y, 0.75, 0.34, f"$\\mathbf{{k}}$ {t}", h, fs=8)
        ax.plot([4.8, 5.2], [1.94, y + 0.17], color=C["red"], lw=0.5 + 4 * w[k], alpha=0.35 + 0.65 * w[k])
        ax.add_patch(Rectangle((6.05, y + 0.06), 0.9 * w[k] / w.max(), 0.22, fc=C["red"], ec="none", alpha=0.85))
        label(ax, C, 7.05, y + 0.17, f"{w[k]:.2f}", fs=8, color=C.ink, ha="left")
    label(ax, C, 6.5, 2.98, "softmax weights", fs=8, color=C["red"])
    # blended output
    x = 5.2
    for k, h in enumerate(hues):
        ax.add_patch(Rectangle((x, 0.55), 2.2 * w[k], 0.34, fc=C.soft(h, 0.7), ec=C.surface, lw=1))
        x += 2.2 * w[k]
    label(ax, C, 5.1, 0.72, "output", fs=9, color=C.ink, ha="right")
    label(ax, C, 6.3, 0.3, "$\\sum_j a_j\\mathbf{v}_j$: mostly 'east', some 'yard'", fs=9, color=C.ink)
    return fig


@figure
def attention_masks(C):
    fig, axs = plt.subplots(1, 4, figsize=(8, 2.75), gridspec_kw=dict(wspace=0.5))
    n = 6
    causal = np.tril(np.ones((n, n)))
    pad = np.ones((n, n))
    pad[:, 4:] = 0
    prefix = np.tril(np.ones((n, n)))
    prefix[:3, :3] = 1
    block = np.zeros((n, n))
    block[:, :4] = 1
    block[:4, 4:] = 0
    block[4:, 4:] = 1
    specs = [("causal (decoder)", causal, ["Survey", "the", "east", "yard", ",", "find"]),
             ("padding", pad, ["find", "the", "red", "box", "<pad>", "<pad>"]),
             ("prefix-LM", prefix, ["img", "img", "txt", "out", "out", "out"]),
             ("block (π0-style)", block, ["img", "img", "txt", "txt", "act", "act"])]
    for ax, (t, M, toks) in zip(axs, specs):
        rgb = np.where(M[..., None] > 0, np.array(to_rgb(C["blue"])), np.array(mix(C.faint, C.surface, 0.35)))
        ax.imshow(rgb)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=C.surface, lw=1.5)
        ax.tick_params(which="both", length=0)
        ax.set_xticks(range(n), toks, rotation=90, fontsize=7.5)
        ax.set_yticks(range(n), toks, fontsize=7.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=9.5)
    axs[0].set_ylabel("query $i$ (reads)", fontsize=8.5)
    fig.text(0.5, -0.1, "key $j$ (is read).  Filled = may attend ($M_{ij}=0$); empty = masked ($M_{ij}=-\\infty$).",
             ha="center", fontsize=9, color=C.muted)
    return fig


@figure
def scaling_saturation(C):
    rng = np.random.default_rng(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.8), gridspec_kw=dict(wspace=0.3))
    for d_k, hue in [(4, "blue"), (32, "aqua"), (128, "orange")]:
        s = (rng.normal(size=(20000, d_k)) * rng.normal(size=(20000, d_k))).sum(1)
        a.hist(s, bins=120, range=(-40, 40), density=True, histtype="step", lw=1.6, color=C[hue],
               label=f"$d_k={d_k}$: σ = {s.std():.1f}")
    a.set_title("raw scores $\\mathbf{q}\\mathbf{k}^{\\top}$ spread as $\\sqrt{d_k}$")
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.legend(fontsize=7.5, loc="center left")
    dks = 2 ** np.arange(1, 10)
    S = 16
    for scaled, hue, name in [(True, "blue", "divided by $\\sqrt{d_k}$"), (False, "orange", "not scaled")]:
        Hs = []
        for d_k in dks:
            sc = (rng.normal(size=(2000, S, d_k)) * rng.normal(size=(2000, 1, d_k))).sum(-1)
            sc = sc / math.sqrt(d_k) if scaled else sc
            p = np.exp(sc - sc.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            Hs.append(-(p * np.log(p + 1e-12)).sum(-1).mean())
        b.semilogx(dks, Hs, "-o", base=2, color=C[hue], ms=4, label=name)
    b.axhline(math.log(S), color=C.muted, ls="--", lw=1)
    b.text(2, math.log(S) - 0.1, "uniform over 16 keys", fontsize=8, color=C.muted, va="top")
    b.set_xlabel("$d_k$")
    b.set_ylabel("entropy of an attention row")
    b.set_title("unscaled softmax collapses to one-hot")
    b.set_ylim(0, 3.0)
    b.legend(fontsize=8, loc="lower left")
    return fig


@figure
def multi_head_split(C):
    fig, ax = canvas(8, 3.4)
    T, h, dh, s = 5, 4, 2, 0.17
    hues = ["blue", "orange", "aqua", "violet"]
    top = 2.6
    fills = [[C.soft(hues[c // dh], 0.55) for c in range(h * dh)] for _ in range(T)]
    cells(ax, C, 0.15, top, T, h * dh, s, "blue", fills=fills, edge=C.surface)
    label(ax, C, 0.15 + 4 * s, top + 0.2, "$XW_Q$: (T, d)", fs=9, color=C.ink)
    label(ax, C, 0.15 + 4 * s, top - T * s - 0.25, "columns = h groups\nof $d_h=d/h$", fs=8)
    arrow(ax, C, (1.65, top - 0.42), (2.05, top - 0.42))
    label(ax, C, 1.85, top - 0.2, "view +\ntranspose", fs=7.5)
    # stacked heads
    for k in range(h):
        y = top + 0.3 - k * 0.55
        cells(ax, C, 2.15, y, T, dh, s * 0.55, hues[k], fills=[[C.soft(hues[k], 0.55)] * dh] * T, edge=C.surface, lw=0.5)
        label(ax, C, 2.15 + dh * s * 0.55 + 0.1, y - 0.25, f"head {k + 1}", fs=8, ha="left", color=C[hues[k]])
    label(ax, C, 2.4, 0.38, "(h, T, $d_h$)", fs=8.5, color=C.ink)
    arrow(ax, C, (3.05, top - 0.42), (3.45, top - 0.42))
    # per-head attention patterns
    n = 5
    pats = [np.eye(n, k=-1) + np.eye(n) * 0.15, np.zeros((n, n)), np.eye(n), np.tril(np.ones((n, n)))]
    pats[1][:, 0] = 1
    names = ["previous token", "first token (sink)", "itself", "broad, causal"]
    for k, (Pm, nm) in enumerate(zip(pats, names)):
        Pm = np.tril(Pm) + 1e-9
        Pm = Pm / Pm.sum(1, keepdims=True)
        x = 3.55 + (k % 2) * 1.35
        y = top + 0.35 - (k // 2) * 1.3
        cmap = LinearSegmentedColormap.from_list("h", [mix(C.faint, C.surface, 0.3), C[hues[k]]])
        ax.imshow(Pm, cmap=cmap, vmin=0, vmax=1, extent=(x, x + 0.75, y - 0.75, y))
        label(ax, C, x + 0.37, y - 0.9, nm, fs=7.5, color=C[hues[k]])
    label(ax, C, 4.55, 0.38, "h attention maps $A^{(i)}$ (T × T)", fs=8.5, color=C.ink)
    arrow(ax, C, (5.85, top - 0.42), (6.25, top - 0.42))
    cells(ax, C, 6.35, top, T, h * dh, s, "blue", fills=fills, edge=C.surface)
    label(ax, C, 6.35 + 4 * s, top + 0.2, "concat heads", fs=9, color=C.ink)
    arrow(ax, C, (6.35 + 4 * s, top - T * s - 0.05), (6.35 + 4 * s, top - T * s - 0.4))
    label(ax, C, 6.35 + 4 * s + 0.12, top - T * s - 0.22, "$W_O$", fs=10, ha="left", color=C.ink)
    cells(ax, C, 6.35, top - T * s - 0.45, T, h * dh, s, "blue", fills=[[C.soft("blue", 0.25)] * (h * dh)] * T,
          edge=C.surface)
    label(ax, C, 6.35 + 4 * s, 0.38, "(T, d): heads mixed", fs=8.5, color=C.ink)
    return fig


@figure
def cross_attention(C):
    rng = np.random.default_rng(2)
    S, Tq = 12, 4
    ctx = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pick", "red", "tool", "box"]
    logits = rng.normal(size=(Tq, S)) * 0.6
    logits[:, 5:7] += 2.2
    logits[:, 11] += 1.0
    logits[3, 2] += 2.0
    A = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    fig = plt.figure(figsize=(8, 2.9))
    ax = fig.add_axes((0, 0, 0.56, 1))
    W, H = 8 * 0.56, 2.9
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    q = chips(ax, C, 0.75, 2.2, [f"$a_{k + 1}$" for k in range(Tq)], "violet", h=0.3, fs=10, widths=0.5, gap=0.3)
    kv = chips(ax, C, 0.05, 0.6, ctx, ["blue"] * 8 + ["orange"] * 4, h=0.3, fs=7, widths=0.33, gap=0.035)
    for i, a in enumerate(q):
        for j, b in enumerate(kv):
            ax.plot([b, a], [0.9, 2.2], color=C["violet"], lw=4 * A[i, j], alpha=min(1, 0.15 + 1.5 * A[i, j]))
    label(ax, C, 3.95, 2.65, "action queries (T = 4)", fs=9, color=C["violet"], ha="right")
    label(ax, C, 0.05, 0.35, "image patches + instruction tokens from the backbone (S = 12)", fs=8.5, ha="left",
          color=C.ink)
    bx = fig.add_axes((0.6, 0.22, 0.38, 0.62))
    heat(bx, C, A, C.seq("violet"), 0, A.max(), rows=[f"$a_{k + 1}$" for k in range(Tq)], cols=ctx)
    bx.set_title("$A$: T × S, rows sum to 1", fontsize=9)
    return fig


@figure
def gqa(C):
    fig, ax = canvas(8.3, 2.6)
    for p, (name, g, note) in enumerate([("MHA", 8, "KV cache 1×"), ("GQA (g = 2)", 2, "KV cache 1/4"),
                                         ("MQA", 1, "KV cache 1/8")]):
        x0 = 0.5 + p * 2.6
        label(ax, C, x0 + 1.17, 2.4, name, fs=10, color=C.ink, weight="bold")
        qx = [x0 + k * 0.3 for k in range(8)]
        for k, x in enumerate(qx):
            box(ax, C, x, 1.65, 0.24, 0.32, "", "blue", r=0.04)
        kw_ = 2.34 / g
        for j in range(g):
            x = x0 + j * kw_ + kw_ / 2 - 0.14
            box(ax, C, x, 0.65, 0.28, 0.32, "", "orange", r=0.04)
            for k in range(8):
                if k * g // 8 == j:
                    ax.plot([qx[k] + 0.12, x + 0.14], [1.65, 0.97], color=C.muted, lw=0.8)
        label(ax, C, x0 + 1.17, 0.35, note, fs=9, color=C.ink)
    label(ax, C, 0.05, 1.81, "Q", fs=9, color=C["blue"], ha="left")
    label(ax, C, 0.05, 0.81, "K,V", fs=8, color=C["orange"], ha="left")
    return fig


@figure
def sinusoidal_pe(C):
    Tn, d = 64, 64
    t = np.arange(Tn)[:, None]
    i = np.arange(d // 2)[None]
    w = 10000.0 ** (-2 * i / d)
    P = np.zeros((Tn, d))
    P[:, 0::2], P[:, 1::2] = np.sin(t * w), np.cos(t * w)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.3))
    a.imshow(P, cmap=C.div(), vmin=-1, vmax=1, aspect="auto")
    a.set_xlabel("dimension $j$ (pairs: fast → slow)")
    a.set_ylabel("position $t$")
    a.set_title("$p_{t,j}$ for $d=64$")
    for s in a.spines.values():
        s.set_visible(False)
    tt = np.linspace(0, 63, 400)
    for k, hue in [(0, "blue"), (4, "aqua"), (8, "orange"), (16, "violet")]:
        b.plot(tt, np.sin(tt * w[0, k]), color=C[hue], lw=1.5, label=f"pair {k}: $\\omega={w[0, k]:.3g}$")
    b.set_xlabel("position $t$")
    b.set_title("$\\sin(t\\,\\omega_i)$: one clock hand per pair")
    b.legend(fontsize=7.5, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.02))
    b.set_ylim(-1.6, 1.15)
    return fig


@figure
def rope_rotation(C):
    th, phq, phk = 0.3, 0.2, 1.4
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (m, n) in zip(axs, [(3, 1), (6, 4)]):
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=C.faint, lw=1))
        for ph, hue, nm in [(phq, "blue", "q"), (phk, "orange", "k")]:
            ax.annotate("", xy=(math.cos(ph), math.sin(ph)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=C[hue], lw=1, ls="--", alpha=0.5))
        aq, ak = phq + m * th, phk + n * th
        for ang, base, hue, nm, pos in [(aq, phq, "blue", f"$\\tilde q_{{m={m}}}$", m), (ak, phk, "orange", f"$\\tilde k_{{n={n}}}$", n)]:
            ax.annotate("", xy=(math.cos(ang), math.sin(ang)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=C[hue], lw=2.2))
            ax.text(1.25 * math.cos(ang), 1.25 * math.sin(ang), nm, color=C[hue], ha="center", va="center", fontsize=12)
            ax.add_patch(Arc((0, 0), 0.5 + 0.25 * (hue == "orange"), 0.5 + 0.25 * (hue == "orange"),
                             theta1=math.degrees(base), theta2=math.degrees(ang), color=C[hue], lw=1))
        lo, hi = sorted([aq, ak])
        ax.add_patch(Arc((0, 0), 1.45, 1.45, theta1=math.degrees(lo), theta2=math.degrees(hi), color=C.ink, lw=1.6))
        mid = (lo + hi) / 2
        ax.text(0.95 * math.cos(mid), 0.95 * math.sin(mid), f"{abs(aq - ak):.2f} rad", fontsize=8.5,
                ha="center", va="center", color=C.ink)
        ax.set_title(f"query at $m={m}$, key at $n={n}$:  $m-n={m - n}$", fontsize=9.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.1, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.text(0.5, 0.02, "dashed: q and k before rotation.  Rotating q by $m\\theta$ and k by $n\\theta$ changes their angle "
             "by $(m-n)\\theta$ only,\nso the score $\\tilde{\\mathbf{q}}_m\\tilde{\\mathbf{k}}_n^{\\top}$ is the same "
             "in both panels.", ha="center", fontsize=8.5, color=C.muted)
    return fig


@figure
def residual_stream(C):
    fig, ax = canvas(8, 3.0)
    y = 1.0
    ax.add_patch(FancyBboxPatch((0.3, y - 0.13), 7.0, 0.26, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=C.soft("blue", 0.35), ec="none"))
    arrow(ax, C, (7.2, y), (7.75, y), color=C["blue"], lw=2.5)
    label(ax, C, 0.3, y - 0.42, "embeddings $X^{(0)}$", fs=9, color=C.ink, ha="left")
    label(ax, C, 7.75, y - 0.42, "final norm → head", fs=9, color=C.ink, ha="right")
    subs = [("attn", "blue"), ("FFN", "orange"), ("attn", "blue"), ("FFN", "orange")]
    for k, (nm, hue) in enumerate(subs):
        x = 0.9 + k * 1.45
        box(ax, C, x, 1.85, 0.95, 0.5, f"Norm → {nm}", hue, fs=8.5)
        arrow(ax, C, (x + 0.15, y + 0.13), (x + 0.15, 1.85), color=C.muted, lw=1)
        ax.add_patch(Circle((x + 1.05, y), 0.11, fc=C.surface, ec=C.ink, lw=1, zorder=3))
        label(ax, C, x + 1.05, y, "+", fs=10, color=C.ink, zorder=4)
        arrow(ax, C, (x + 0.8, 1.85), (x + 1.05, y + 0.11), color=C[hue], lw=1.3)
        if k == 1:
            label(ax, C, x + 0.5, 2.55, "block 1", fs=8.5)
        if k == 3:
            label(ax, C, x + 0.5, 2.55, "block 2", fs=8.5)
    label(ax, C, 0.9 + 0.15 - 0.05, 1.45, "read", fs=8, ha="right")
    label(ax, C, 0.9 + 0.98, 1.6, "write (add)", fs=8, ha="left", color=C["blue"])
    label(ax, C, 6.8, 2.1, "… L blocks", fs=9, color=C.ink)
    ax.plot([5.2, 5.2], [y - 0.13, 0.35], color=C["violet"], ls=(0, (2, 2)), lw=1.2)
    label(ax, C, 5.3, 0.3, "logit lens: apply the output head here", fs=8.5, color=C["violet"], ha="left")
    label(ax, C, 0.3, 0.3, "the stream ($T\\times d$) is never overwritten, only added to", fs=8.5, ha="left")
    return fig


@figure
def shift_loss_mask(C):
    fig, ax = canvas(8, 2.65)
    seq = ["<img>", "<img>", "…", "In:", "What", "action", "…", "Out:", "a1", "a2", "…", "a7", "</s>"]
    kind = ["blue"] * 3 + ["aqua"] * 5 + ["orange"] * 4 + ["violet"]
    w, gap, x0 = 0.52, 0.06, 0.95
    inp = seq[:-1]
    tgt = seq[1:]
    xs = [x0 + k * (w + gap) for k in range(len(inp))]
    label(ax, C, 0.05, 2.06, "input $x_{\\leq t}$", fs=9, ha="left", color=C.ink)
    label(ax, C, 0.05, 1.11, "target $x_{t+1}$", fs=9, ha="left", color=C.ink)
    label(ax, C, 0.05, 0.42, "loss mask $m_t$", fs=9, ha="left", color=C.ink)
    for k, (a, b) in enumerate(zip(inp, tgt)):
        on = kind[k + 1] in ("orange", "violet")
        box(ax, C, xs[k], 1.9, w, 0.32, a, kind[k], fs=7.5, family="monospace")
        box(ax, C, xs[k], 0.95, w, 0.32, b, kind[k + 1] if on else None, fs=7.5, fill=on,
            tc=C.ink if on else C.muted, ls="-" if on else (0, (2, 2)), family="monospace")
        arrow(ax, C, (xs[k] + w / 2, 1.88), (xs[k] + w / 2, 1.29), color=C.faint, lw=1, ms=8)
        label(ax, C, xs[k] + w / 2, 0.42, "1" if on else "0", fs=9, color=C["orange"] if on else C.muted)
        if not on:
            label(ax, C, xs[k] + w / 2, 0.75, "-100", fs=6.5, color=C.muted)
    label(ax, C, xs[0] + 1.6 * (w + gap), 2.45, "image tokens", fs=8.5, color=C["blue"])
    label(ax, C, xs[4] + w / 2, 2.45, "prompt", fs=8.5, color=C["aqua"])
    label(ax, C, xs[9] + w / 2, 2.45, "7 action tokens", fs=8.5, color=C["orange"])
    label(ax, C, 0.05, 1.6, "each position\npredicts the next", fs=7.5, ha="left")
    return fig


@figure
def prefill_decode(C):
    fig, ax = canvas(8, 2.9)
    y = 1.85
    box(ax, C, 0.2, y, 2.5, 0.5, "prefill: image + prompt,\n~280 tokens in one parallel pass", "blue", fs=8.5)
    label(ax, C, 1.45, y + 0.72, "compute-bound", fs=8.5, color=C["blue"])
    x = 2.8
    for k in range(7):
        box(ax, C, x, y, 0.62, 0.5, f"$a_{k + 1}$", "orange", fs=10)
        x += 0.68
    label(ax, C, 2.8 + 3.5 * 0.68, y + 0.72, "decode: one token per pass — memory-bandwidth-bound", fs=8.5, color=C["orange"])
    arrow(ax, C, (0.2, 1.55), (7.7, 1.55), color=C.muted, lw=1)
    label(ax, C, 7.75, 1.55, "time", fs=8.5, ha="left")
    # KV cache growth
    label(ax, C, 0.2, 1.2, "KV cache", fs=8.5, ha="left", color=C.ink)
    box(ax, C, 0.2, 0.65, 2.5, 0.3, "280 entries per layer", "violet", fs=7.5, r=0.03)
    x = 2.8
    for k in range(7):
        box(ax, C, x, 0.65, 0.62, 0.3, f"+{k + 1}", "violet", fs=7.5, r=0.03)
        x += 0.68
    label(ax, C, 0.2, 0.25, "every decode step re-reads all weights and the whole cache, to produce one token", fs=8.5,
          ha="left")
    return fig


@figure
def roofline(C):
    bw, peak = 1.0e12, 150e12  # illustrative GPU: 1 TB/s, 150 TFLOP/s
    I = np.logspace(-1, 4, 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.loglog(I, np.minimum(peak, bw * I) / 1e12, color=C.ink, lw=2)
    ridge = peak / bw
    ax.axvline(ridge, color=C.faint, ls="--", lw=1)
    ax.text(ridge * 1.1, 0.25, f"ridge: {ridge:.0f} FLOP/byte", fontsize=8, color=C.muted)
    ax.text(0.12, 40, "memory-bound:\nspeed = bandwidth × intensity", fontsize=8.5, color=C.muted)
    ax.text(900, peak / 1e12 * 1.25, "compute-bound", fontsize=8.5, color=C.muted, ha="center")
    for x, t, hue in [(1, "decode,\nbatch 1", "orange"), (8, "decode, batch 8\n(8 robots, one GPU)", "yellow"),
                      (280, "prefill,\n280 tokens", "blue")]:
        yv = min(peak, bw * x) / 1e12
        ax.plot(x, yv, "o", color=C[hue], ms=9, mec=C.surface, mew=1.5, zorder=3)
        ax.text(x * 1.3, yv * 0.42, t, fontsize=8.5, color=C.ink, va="top")
    ax.set_xlabel("arithmetic intensity (FLOPs per byte moved)")
    ax.set_ylabel("attainable TFLOP/s")
    ax.set_title("roofline of an illustrative GPU (1 TB/s, 150 TFLOP/s)", fontsize=9.5)
    ax.set_ylim(0.05, 600)
    ax.grid(True, which="major", alpha=0.6)
    return fig


@figure
def scaling_laws(C):
    E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
    L = lambda N, D: E + A / N ** al + B / D ** be  # noqa: E731
    Ns = np.logspace(7, 12, 400)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    budgets = [1e19, 1e20, 1e21, 1e22, 1e23, 1e24]
    cmap = C.seq("blue")
    best = []
    for k, Cb in enumerate(budgets):
        D = Cb / (6 * Ns)
        loss = L(Ns, D)
        ok = D > 1e8
        col = cmap(0.3 + 0.7 * k / (len(budgets) - 1))
        ax.semilogx(Ns[ok], loss[ok], color=col, lw=1.6)
        i = np.argmin(loss)
        best.append((Ns[i], loss[i], D[i]))
        ax.text(Ns[i], loss[i] - 0.06, f"$10^{{{int(math.log10(Cb))}}}$", fontsize=8, color=C.ink, ha="center", va="top")
    bN, bL, bD = np.array(best).T
    ax.plot(bN, bL, "o--", color=C["orange"], ms=5, lw=1.2)
    r = bD / bN
    ax.text(3e10, 3.55, f"compute-optimal points (orange);\nthis fit gives D/N = {r.min():.0f}–{r.max():.0f}"
            " tokens\nper parameter over this range", fontsize=8.5, color=C["orange"], va="top")
    ax.axhline(E, color=C.muted, ls=":", lw=1)
    ax.text(1.2e7, E + 0.02, "irreducible loss E", fontsize=8, color=C.muted, va="bottom")
    ax.set_xlabel("parameters N")
    ax.set_ylabel("predicted loss $\\hat{\\mathcal{L}}(N, D=C/6N)$")
    ax.set_title("Chinchilla fit: one curve per compute budget C (FLOPs, labelled at its minimum)", fontsize=9.5)
    ax.set_ylim(1.6, 4.0)
    return fig


# ======================================================================= Chapter 3

def _scene_img(res=96, seed=4, n=2, size=0.24, texture=0.0):
    import random as _r
    from vlacourse.data import scenes as S_
    return S_.render(S_.random_scene(_r.Random(seed), n_objects=n), res, size=size, texture=texture)


def _explicit(objs, res=128, size=0.2):
    from vlacourse.data import scenes as S_
    return S_.render(S_.Scene(tuple(S_.Obj(*o) for o in objs)), res, size=size, texture=0.0)


DEPOT3 = [("toolbox", "red", 0.68, 0.72), ("crate", "blue", 0.78, 0.25), ("barrel", "yellow", 0.22, 0.45)]


@figure
def conv_sliding(C):
    fig, ax = canvas(8, 3.2)
    s, n = 0.3, 7
    rng = np.random.default_rng(3)
    vals = rng.integers(0, 10, (n, n))
    top = 2.75
    cells(ax, C, 0.2, top, n, n, s, "blue", fills=[[C.soft("blue", 0.08 + 0.05 * v) for v in r] for r in vals],
          texts=[[str(v) for v in r] for r in vals], fs=7)
    wins = [((0, 0), "orange"), ((3, 2), "violet")]
    for (r, c), hue in wins:
        ax.add_patch(Rectangle((0.2 + c * s, top - (r + 3) * s), 3 * s, 3 * s, fc="none", ec=C[hue], lw=2.4))
    label(ax, C, 0.2 + n * s / 2, 0.45, "input X (one channel, 7 × 7)", fs=9, color=C.ink)
    K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    cells(ax, C, 3.35, top - 0.6, 3, 3, s, "aqua", fills=[[C.soft("aqua", 0.3)] * 3] * 3,
          texts=[[str(v) for v in r] for r in K], fs=8)
    label(ax, C, 3.35 + 1.5 * s, top - 0.38, "kernel K (3 × 3)", fs=9, color=C.ink)
    label(ax, C, 3.35 + 1.5 * s, 0.85, "the same 9 weights\nat every position", fs=8.5, color=C["aqua"])
    out = np.array([[(vals[i:i + 3, j:j + 3] * K).sum() for j in range(5)] for i in range(5)])
    ox = 5.6
    cmap, norm = C.div(), plt.Normalize(-30, 30)
    cells(ax, C, ox, top - 0.3, 5, 5, s, "violet", fills=[[cmap(norm(v)) for v in r] for r in out],
          texts=[[str(v) for v in r] for r in out], fs=7, edge=C.surface)
    for (r, c), hue in wins:
        ax.add_patch(Rectangle((ox + c * s, top - 0.3 - (r + 1) * s), s, s, fc="none", ec=C[hue], lw=2.4))
        arrow(ax, C, (0.2 + (c + 3) * s, top - (r + 1.5) * s), (ox + c * s, top - 0.3 - (r + 0.5) * s),
              color=C[hue], lw=1.1, rad=-0.15)
    label(ax, C, ox + 2.5 * s, 0.45, "output Y (5 × 5): $Y_{ij}=\\sum_{u,v}K_{uv}X_{i+u,j+v}$", fs=9, color=C.ink)
    return fig


@figure
def conv_equivariance(C):
    img = _scene_img(96, seed=4, n=2).astype(float).mean(-1) / 255
    K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)

    def conv(x):
        xp = np.pad(x, 1, mode="wrap")
        return sum(K[u, v] * xp[u:u + x.shape[0], v:v + x.shape[1]] for u in range(3) for v in range(3))

    sh = np.roll(img, 12, axis=1)
    fig, axs = plt.subplots(1, 4, figsize=(8, 2.4), gridspec_kw=dict(wspace=0.08))
    lim = np.abs(conv(img)).max()
    for ax, im_, t, cm in [(axs[0], img, "image $X$", "gray"), (axs[1], conv(img), "edge response $f(X)$", None),
                           (axs[2], sh, "shifted $T_{12}X$", "gray"), (axs[3], conv(sh), "$f(T_{12}X)=T_{12}f(X)$", None)]:
        ax.imshow(im_, cmap=cm or C.div(), vmin=None if cm else -lim, vmax=None if cm else lim)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    err = np.abs(conv(sh) - np.roll(conv(img), 12, axis=1)).max()
    fig.text(0.5, -0.02, f"max |f(shifted) − shifted f| = {err:.1e}    "
             f"pooled |response|: {np.abs(conv(img)).mean():.4f} vs {np.abs(conv(sh)).mean():.4f} (invariant)",
             ha="center", fontsize=8.5, color=C.muted)
    return fig


@figure
def receptive_field(C):
    fig, ax = canvas(7.4, 3.1)
    n, s = 11, 0.42
    x0 = 0.5
    rows = [(0.45, "input"), (1.15, "layer 1"), (1.85, "layer 2"), (2.55, "layer 3")]
    paths = np.zeros((4, n))
    paths[3, n // 2] = 1
    for l in (2, 1, 0):
        for i in range(n):
            paths[l, i] = sum(paths[l + 1, j] for j in (i - 1, i, i + 1) if 0 <= j < n)
    for l, (y, name) in enumerate(rows):
        for i in range(n):
            on = paths[l, i] > 0
            fc = C.soft("orange", 0.12 + 0.8 * paths[l, i] / paths[l].max()) if on else C.soft("blue", 0.06)
            ax.add_patch(Rectangle((x0 + i * s, y), s * 0.9, 0.36, fc=fc, ec=C["orange"] if on else C.faint, lw=1))
            if l == 0 and on:
                label(ax, C, x0 + i * s + s * 0.45, y + 0.18, f"{int(paths[l, i])}", fs=8, color=C.ink)
        label(ax, C, x0 - 0.1, y + 0.18, name, fs=8.5, ha="right", color=C.ink)
        if l < 3:
            for i in range(n):
                if paths[l + 1, i] > 0:
                    for j in (i - 1, i, i + 1):
                        ax.plot([x0 + i * s + s * 0.45, x0 + j * s + s * 0.45], [y + 0.7, y + 0.36],
                                color=C["orange"], lw=0.6, alpha=0.6)
    w = int((paths[0] > 0).sum())
    label(ax, C, x0 + n * s + 0.15, 0.63, f"receptive field: {w} pixels = 2L + 1\nnumbers: paths to the output unit\n"
          "(the effective receptive field)", fs=8.5, ha="left", color=C.ink)
    return fig


@figure
def vit_architecture(C):
    fig, ax = canvas(8, 3.5)
    img = _scene_img(96, seed=7, n=2, size=0.24)
    g = 4
    ax.imshow(img, extent=(0.15, 1.65, 1.0, 2.5), interpolation="bilinear")
    for k in range(g + 1):
        ax.plot([0.15 + k * 1.5 / g] * 2, [1.0, 2.5], color="white", lw=1.2)
        ax.plot([0.15, 1.65], [1.0 + k * 1.5 / g] * 2, color="white", lw=1.2)
    label(ax, C, 0.9, 0.75, "image, P × P patches", fs=8.5, color=C.ink)
    arrow(ax, C, (1.75, 1.75), (2.15, 1.75))
    label(ax, C, 1.95, 2.0, "$\\cdot E$", fs=10, color=C.ink)
    # token column
    xs = 2.3
    box(ax, C, xs, 2.75, 0.5, 0.26, "[CLS]", "orange", fs=7, r=0.04)
    for k in range(6):
        box(ax, C, xs, 2.38 - k * 0.3, 0.5, 0.24, "" if k != 5 else "⋮", "blue", fs=8, r=0.04)
    label(ax, C, xs + 0.25, 0.55, "1 + $N_p$ tokens\n+ $E_\\mathrm{pos}$", fs=8.5, color=C.ink)
    arrow(ax, C, (2.9, 1.75), (3.3, 1.75))
    box(ax, C, 3.35, 0.9, 1.6, 1.9, "L × block\n\nbidirectional\nself-attention\n+ MLP", "violet", fs=9)
    arrow(ax, C, (5.0, 1.75), (5.4, 1.75))
    xo = 5.45
    box(ax, C, xo, 2.75, 0.5, 0.26, "", "orange", r=0.04)
    for k in range(6):
        box(ax, C, xo, 2.38 - k * 0.3, 0.5, 0.24, "" if k != 5 else "⋮", "blue", fs=8, r=0.04)
    arrow(ax, C, (6.0, 2.88), (6.35, 2.88), color=C["orange"])
    label(ax, C, 6.4, 2.88, "[CLS] output: one vector\n(classification, CLIP)", fs=8.5, ha="left", color=C.ink)
    arrow(ax, C, (6.0, 1.65), (6.35, 1.65), color=C["blue"])
    label(ax, C, 6.4, 1.65, "patch features $N_p\\times d$:\none per region, they\nkeep where (VLM projector)", fs=8.5,
          ha="left", color=C.ink)
    return fig


@figure
def resolution_tokens(C):
    res = np.arange(112, 1025, 16)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw=dict(wspace=0.3))
    d, L = 768, 12
    for P, hue in [(14, "blue"), (16, "aqua"), (32, "orange")]:
        N = (res // P) ** 2
        a.semilogy(res, N, color=C[hue], label=f"P = {P}")
        fl = (2 * (N + 1) * 12 * L * d ** 2 + 4 * L * (N + 1) ** 2 * d) / 1e9
        b.semilogy(res, fl, color=C[hue], label=f"ViT-B/{P}")
    for ax in (a, b):
        ax.set_xlabel("image side (pixels)")
        ax.grid(True, which="major", alpha=0.5)
        ax.legend(fontsize=8)
    a.set_title("patch tokens per image")
    b.set_title("encoder GFLOPs per image")
    a.axvline(224, color=C.faint, ls="--", lw=1)
    b.axvline(224, color=C.faint, ls="--", lw=1)
    return fig


@figure
def dino(C):
    fig, ax = canvas(8, 3.6)
    img = _scene_img(120, seed=11, n=3, size=0.2)
    ax.imshow(img, extent=(0.1, 1.9, 0.9, 2.7))
    for (x, y, w, h), hue in [((0.15, 1.0, 1.4, 1.5), "blue"), ((0.4, 1.15, 1.45, 1.4), "blue"),
                              ((0.3, 2.0, 0.5, 0.5), "orange"), ((1.2, 1.1, 0.45, 0.45), "orange"),
                              ((0.95, 1.85, 0.5, 0.5), "orange")]:
        ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=C[hue], lw=1.4))
    label(ax, C, 1.0, 0.6, "2 global crops (blue)\nseveral local crops (orange)", fs=8, color=C.ink)
    box(ax, C, 2.6, 2.35, 1.4, 0.55, "teacher $g_{\\theta_t}$\n(global crops only)", "blue", fs=8.5)
    box(ax, C, 2.6, 0.75, 1.4, 0.55, "student $g_{\\theta_s}$\n(all crops)", "orange", fs=8.5)
    arrow(ax, C, (1.95, 2.3), (2.6, 2.6), color=C["blue"])
    arrow(ax, C, (1.95, 1.5), (2.6, 1.0), color=C["orange"])
    box(ax, C, 4.4, 2.35, 1.6, 0.55, "softmax$((g-\\mathbf{c})/\\tau_t)$\ncenter + sharpen", "blue", fs=8)
    box(ax, C, 4.4, 0.75, 1.6, 0.55, "softmax$(g/\\tau_s)$", "orange", fs=8.5)
    arrow(ax, C, (4.0, 2.62), (4.4, 2.62), color=C["blue"])
    arrow(ax, C, (4.0, 1.02), (4.4, 1.02), color=C["orange"])
    box(ax, C, 6.4, 1.55, 1.5, 0.55, "$H(P_t, P_s)$\ncross-entropy", "violet", fs=9)
    arrow(ax, C, (6.0, 2.62), (6.9, 2.1), color=C["blue"])
    arrow(ax, C, (6.0, 1.02), (6.9, 1.55), color=C["orange"])
    label(ax, C, 6.6, 2.55, "stop-gradient", fs=8, color=C["blue"])
    arrow(ax, C, (3.3, 1.3), (3.3, 2.35), color=C.muted, ls=(0, (3, 2)))
    label(ax, C, 3.4, 1.82, "EMA: $\\theta_t\\leftarrow\\lambda\\theta_t+(1-\\lambda)\\theta_s$", fs=8.5, ha="left", color=C.ink)
    label(ax, C, 5.5, 0.35, "gradients flow only into the student", fs=8.5, color=C["orange"])
    return fig


@figure
def mae(C):
    fig, ax = canvas(8, 2.9)
    img = _scene_img(96, seed=7, n=2, size=0.26)
    g = 6
    rng = np.random.default_rng(0)
    keep = set(rng.choice(g * g, g * g // 4, replace=False).tolist())
    st = 1.6 / g

    def grid(x0, masked_fill, show_masked):
        ax.imshow(img, extent=(x0, x0 + 1.6, 0.8, 2.4))
        for k in range(g * g):
            r, c = divmod(k, g)
            if k not in keep:
                ax.add_patch(Rectangle((x0 + c * st, 2.4 - (r + 1) * st), st, st,
                                       fc=masked_fill, ec=C.surface, lw=0.8, alpha=1 if not show_masked else 0.0))
                if show_masked:
                    ax.add_patch(Rectangle((x0 + c * st, 2.4 - (r + 1) * st), st, st, fc="none",
                                           ec=C["orange"], lw=0.8))

    grid(0.15, mix(C.faint, C.surface, 0.7), False)
    label(ax, C, 0.95, 0.5, "input: 75% of patches\nremoved", fs=8.5, color=C.ink)
    arrow(ax, C, (1.85, 1.6), (2.2, 1.6))
    xs = [2.3 + k * 0.17 for k in range(len(keep))]
    for x in xs:
        box(ax, C, x, 1.45, 0.13, 0.3, "", "blue", r=0.02, lw=0.8)
    box(ax, C, 2.25, 1.95, 1.6, 0.4, "ViT encoder", "blue", fs=9)
    label(ax, C, 3.05, 1.2, f"only the {len(keep)} visible\npatches (cheap)", fs=8, color=C.ink)
    arrow(ax, C, (3.95, 1.6), (4.3, 1.6))
    box(ax, C, 4.35, 1.3, 1.25, 0.6, "small decoder\n+ mask tokens", "violet", fs=8.5)
    arrow(ax, C, (5.65, 1.6), (6.05, 1.6))
    grid(6.15, "none", True)
    label(ax, C, 6.95, 0.5, "target: pixels of the hidden\npatches (orange); loss only there", fs=8.5, color=C.ink)
    return fig


@figure
def token_reduction(C):
    fig, ax = canvas(8, 3.1)
    s = 0.17

    def grid(x, y, n, sz, hue, alpha=0.3):
        for r in range(n):
            for c in range(n):
                ax.add_patch(Rectangle((x + c * sz, y - (r + 1) * sz), sz, sz, fc=C.soft(hue, alpha), ec=C[hue], lw=0.6))

    for p, t in enumerate(["pooling 2×2", "pixel shuffle r = 2", "learned resampler", "crop"]):
        label(ax, C, 0.78 + p * 2.0, 2.95, t, fs=9.5, color=C.ink, weight="bold")
    top, low = 2.75, 1.2
    # pooling: 8x8 -> 4x4 cells of double size
    grid(0.1, top, 8, s, "blue")
    grid(0.1, low, 4, s * 2 * 0.55, "blue", 0.55)
    label(ax, C, 0.78, 0.25, "64 → 16 tokens, width d\n(neighbours averaged)", fs=7.5, color=C.ink)
    # pixel shuffle: 16 tokens, each 4 neighbours wide
    hues = ["blue", "aqua", "orange", "violet"]
    for r in range(8):          # colour = position inside each 2x2 block
        for c in range(8):
            h_ = hues[(r % 2) * 2 + c % 2]
            ax.add_patch(Rectangle((2.1 + c * s, top - (r + 1) * s), s, s, fc=C.soft(h_, 0.45), ec=C.surface, lw=0.6))
    for r in range(4):
        for c in range(4):
            for k, h in enumerate(hues):
                ax.add_patch(Rectangle((2.1 + c * 0.38 + k * 0.085, low - (r + 1) * 0.15), 0.085, 0.15,
                                       fc=C.soft(h, 0.55), ec=C.surface, lw=0.4))
    label(ax, C, 2.78, 0.25, "64 → 16 tokens, width 4d\n(neighbours concatenated)", fs=7.5, color=C.ink)
    # resampler
    grid(4.1, top, 8, s, "blue")
    for q in range(8):
        x = 4.12 + q * 0.17
        box(ax, C, x, 0.75, 0.13, 0.3, "", "violet", r=0.02, lw=0.8)
        for k in range(q % 3, 64, 7):
            r, c = divmod(k, 8)
            ax.plot([x + 0.065, 4.1 + (c + .5) * s], [1.05, top - (r + 1) * s], color=C["violet"], lw=0.3, alpha=0.45)
    label(ax, C, 4.78, 0.25, "64 → 8 tokens\n(learned queries cross-attend)", fs=7.5, color=C.ink)
    # crop around the red object
    img = _scene_img(96, seed=4, n=2, size=0.24)
    red = (img[..., 0] > 150) & (img[..., 1] < 90)
    cy, cx = [int(v.mean()) for v in np.nonzero(red)]
    h = 18
    r0, c0 = max(0, cy - h), max(0, cx - h)
    x0, y0, side = 6.1, top - 1.36, 1.36
    ax.imshow(img, extent=(x0, x0 + side, y0, y0 + side))
    sc_ = side / 96
    ax.add_patch(Rectangle((x0 + c0 * sc_, y0 + side - (r0 + 2 * h) * sc_), 2 * h * sc_, 2 * h * sc_, fc="none",
                           ec=C["orange"], lw=1.8))
    ax.imshow(img[r0:r0 + 2 * h, c0:c0 + 2 * h], extent=(6.45, 7.05, low - 0.6, low))
    label(ax, C, 6.78, 0.25, "spend tokens only\nwhere it matters", fs=7.5, color=C.ink)
    return fig


# ======================================================================= Chapter 4

@figure
def contrastive_matrix(C):
    rng = np.random.default_rng(3)
    B, d = 6, 16
    u = rng.normal(size=(B, d))
    v = u + 0.9 * rng.normal(size=(B, d))
    v[2] = 0.6 * v[2] + 0.8 * u[4]          # a hard negative pair
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    S = u @ v.T / 0.1
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    sig = 1 / (1 + np.exp(-(S * 1.0 - 4)))
    names_i = [f"img {k + 1}" for k in range(B)]
    names_t = [f"cap {k + 1}" for k in range(B)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.5), gridspec_kw=dict(wspace=0.45))
    heat(a, C, P, C.seq("blue"), 0, 1, rows=names_i, cols=names_t, annot="{:.2f}", fs=7)
    a.set_title("InfoNCE: softmax over each row\n(target: the diagonal)", fontsize=9.5)
    heat(b, C, sig, C.seq("violet"), 0, 1, rows=names_i, cols=names_t, annot="{:.2f}", fs=7)
    b.set_title("SigLIP: independent sigmoid per cell\n(target: 1 on diagonal, 0 elsewhere)", fontsize=9.5)
    for ax in (a, b):
        for k in range(B):
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fc="none", ec=C["orange"], lw=1.8))
    return fig


@figure
def contrastive_geometry(C):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=C.faint, lw=1.2))
    pol = lambda deg: np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])  # noqa: E731
    pu = pol(15)
    ax.plot(*pu, "s", color=C["blue"], ms=10, zorder=3)
    ax.text(*(pu * 1.1 + np.array([0.02, -0.12])), "$\\mathbf{u}_i$ (image)", color=C["blue"], ha="left", fontsize=10)
    caps = [(45, "red toolbox", "orange", None), (85, "blue toolbox", "violet", 0.55), (160, "green barrel", "aqua", 0.12),
            (250, "yellow cone", "yellow", 0.02)]
    for ang, name, hue, p in caps:
        pv = pol(ang)
        ax.plot(*pv, "o", color=C[hue], ms=8, zorder=3)
        ax.text(*(pv * 1.17), f"'{name}'", color=C[hue], ha="center", va="center", fontsize=9)
        if p is None:
            ax.annotate("", xy=pu + 0.75 * (pv - pu), xytext=pu, arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=2.2))
            ax.text(0.55, 0.47, "pull toward\nthe positive", color=C["orange"], fontsize=8.5, ha="center")
        elif p > 0.05:
            d = (pu - pv) / np.linalg.norm(pu - pv)
            tip = pu + d * 0.55 * p
            ax.annotate("", xy=tip, xytext=pu, arrowprops=dict(arrowstyle="-|>", color=C[hue], lw=1 + 3 * p))
            ax.plot([pv[0], pu[0]], [pv[1], pu[1]], color=C[hue], lw=0.6, ls=(0, (2, 3)))
        else:
            ax.plot([pv[0], pu[0]], [pv[1], pu[1]], color=C.faint, lw=0.6, ls=(0, (2, 3)))
    ax.text(0, -0.12, "push away from each negative,\nweighted by its softmax probability:\n"
            "'blue toolbox' $p_{ij}$=0.55 (hard), 'green barrel' 0.12,\n'yellow cone' 0.02 (ignored)",
            ha="center", va="center", fontsize=8.5, color=C.ink)
    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-1.3, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


@figure
def clip_zero_shot(C):
    fig, ax = canvas(8, 3.3)
    img = _explicit([("toolbox", "red", 0.45, 0.55), ("crate", "blue", 0.8, 0.2)], 96, size=0.3)
    # training side
    label(ax, C, 1.75, 3.1, "training: match pairs", fs=9.5, color=C.ink, weight="bold")
    ax.imshow(img, extent=(0.15, 1.05, 1.9, 2.8))
    box(ax, C, 1.3, 1.95, 1.0, 0.36, "image tower", "blue", fs=8.5)
    box(ax, C, 0.1, 1.1, 1.05, 0.5, "'a red toolbox\nnear a crate'", "orange", fs=7.5)
    box(ax, C, 1.3, 1.17, 1.0, 0.36, "text tower", "orange", fs=8.5)
    arrow(ax, C, (1.05, 2.13), (1.3, 2.13))
    arrow(ax, C, (1.15, 1.35), (1.3, 1.35))
    ax.add_patch(Circle((3.0, 1.75), 0.55, fill=False, ec=C.faint))
    ax.plot(3.0 + 0.55 * math.cos(1.0), 1.75 + 0.55 * math.sin(1.0), "s", color=C["blue"], ms=7)
    ax.plot(3.0 + 0.55 * math.cos(1.2), 1.75 + 0.55 * math.sin(1.2), "o", color=C["orange"], ms=7)
    arrow(ax, C, (2.3, 2.13), (2.75, 2.2), color=C["blue"], lw=1)
    arrow(ax, C, (2.3, 1.35), (2.7, 1.5), color=C["orange"], lw=1)
    label(ax, C, 3.0, 1.0, "joint space $\\mathbb{R}^{d_e}$", fs=8.5, color=C.ink)
    # zero-shot side
    ax.plot([3.85, 3.85], [0.3, 3.1], color=C.faint, lw=1)
    label(ax, C, 6.0, 3.1, "zero-shot: the classifier is made of text", fs=9.5, color=C.ink, weight="bold")
    classes = ["red toolbox", "blue crate", "yellow cone", "green barrel"]
    sims = [0.31, 0.22, 0.18, 0.16]
    for k, (c_, s_) in enumerate(zip(classes, sims)):
        y = 2.55 - k * 0.48
        box(ax, C, 4.05, y, 2.1, 0.36, f"'a top-down photo of a {c_}'", "orange", fs=7)
        label(ax, C, 6.35, y + 0.18, "→ $\\mathbf{v}_" + str(k + 1) + "$", fs=9, color=C.ink)
        hit = k == 0
        ax.add_patch(Rectangle((6.75, y + 0.06), 1.1 * s_ / max(sims), 0.24, fc=C["blue"] if hit else C.faint, ec="none"))
        label(ax, C, 6.8 + 1.1 * s_ / max(sims), y + 0.18, f"{s_:.2f}", fs=8, color=C.ink, ha="left")
    label(ax, C, 7.25, 0.45, "cosine with the image\n→ argmax = class", fs=8.5, color=C.ink)
    return fig


@figure
def connectors(C):
    fig, ax = canvas(8, 4.2)

    def tokens(x, y, n, hue, w=0.12, h=0.28, gap=0.03):
        for k in range(n):
            box(ax, C, x + k * (w + gap), y, w, h, "", hue, r=0.02, lw=0.7)
        return x + n * (w + gap)

    rows = [("projector (LLaVA, OpenVLA, π0)", 16, "MLP per token", 16, "all patches"),
            ("pixel shuffle + linear (SmolVLM)", 16, "concat 2×2, linear", 4, "fewer, wider"),
            ("resampler / Q-Former (BLIP-2)", 16, "learned queries\ncross-attend", 3, "fixed number")]
    for r, (name, n_in, op, n_out, note) in enumerate(rows):
        y = 3.55 - r * 0.95
        label(ax, C, 0.05, y + 0.48, name, fs=9, color=C.ink, ha="left", weight="bold")
        xe = tokens(0.1, y, n_in, "blue")
        arrow(ax, C, (xe + 0.05, y + 0.14), (xe + 0.4, y + 0.14))
        box(ax, C, xe + 0.45, y - 0.04, 1.25, 0.36, op, "violet", fs=7.5)
        arrow(ax, C, (xe + 1.75, y + 0.14), (xe + 2.1, y + 0.14))
        w = 0.12 if r != 1 else 0.42
        xo = tokens(xe + 2.15, y, n_out, "orange", w=w)
        box(ax, C, xo + 0.15, y - 0.04, 1.0, 0.36, "LLM input", "aqua", fs=8)
        label(ax, C, xo + 1.3, y + 0.14, note, fs=8, ha="left", color=C.muted)
    # Flamingo
    y = 0.5
    label(ax, C, 0.05, y + 0.95, "gated cross-attention (Flamingo)", fs=9, color=C.ink, ha="left", weight="bold")
    xe = tokens(0.1, y + 0.2, 16, "blue")
    for k in range(3):
        x = 3.2 + k * 1.45
        box(ax, C, x, y + 0.48, 1.1, 0.3, "frozen LM layer", "aqua", fs=7.5)
        box(ax, C, x, y - 0.02, 1.1, 0.3, "tanh(α)·xattn", "violet", fs=7.5)
        arrow(ax, C, (x + 0.55, y + 0.28), (x + 0.55, y + 0.48), lw=1)
        if k < 2:
            arrow(ax, C, (x + 1.1, y + 0.63), (x + 1.45, y + 0.13), lw=1)
    ax.annotate("", xy=(3.2, y + 0.13), xytext=(xe + 0.05, y + 0.34), arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1))
    label(ax, C, 6.9, y - 0.2, "image read inside the LM;\nα = 0 at start: identity", fs=8, ha="left", color=C.muted)
    return fig


@figure
def grounding_map(C):
    img = _explicit(DEPOT3)
    g = 8
    P = img.reshape(g, 128 // g, g, 128 // g, 3).mean((1, 3)) / 255
    red = np.clip(P[..., 0] - np.maximum(P[..., 1], P[..., 2]) - 0.05, 0, None)
    sim = 0.05 + 0.6 * red / red.max()
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.8), gridspec_kw=dict(wspace=0.3))
    axs[0].imshow(img)
    axs[0].set_title("image (8 × 8 patches)", fontsize=9)
    for k in range(g + 1):
        axs[0].axhline(k * 16 - 0.5, color="white", lw=0.6)
        axs[0].axvline(k * 16 - 0.5, color="white", lw=0.6)
    heat(axs[1], C, sim, C.seq("orange"), 0, 0.7, annot="{:.1f}", fs=6)
    axs[1].set_title("cos(patch, 'the red toolbox')", fontsize=9)
    axs[2].imshow(img)
    axs[2].imshow(sim, cmap=C.seq("orange"), alpha=0.6, extent=(-0.5, 127.5, 127.5, -0.5), interpolation="bilinear")
    r, c = np.unravel_index(sim.argmax(), sim.shape)
    axs[2].plot(c * 16 + 8, r * 16 + 8, "x", color=C.ink, ms=10, mew=2)
    axs[2].set_title("upsampled heatmap, peak = point", fontsize=9)
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_visible(False)
    return fig


@figure
def loc_tokens(C):
    fig, ax = canvas(7.8, 2.7)
    img = _explicit(DEPOT3)
    x0, y0, side = 0.2, 0.3, 2.2
    ax.imshow(img, extent=(x0, x0 + side, y0, y0 + side))
    red = (img[..., 0] > 150) & (img[..., 1] < 90)
    rows, cols = np.nonzero(red)
    ymin, ymax, xmin, xmax = rows.min() / 128, (rows.max() + 1) / 128, cols.min() / 128, (cols.max() + 1) / 128
    ax.add_patch(Rectangle((x0 + xmin * side, y0 + (1 - ymax) * side), (xmax - xmin) * side, (ymax - ymin) * side,
                           fc="none", ec=C["orange"], lw=2))
    box_ = [ymin, xmin, ymax, xmax]
    toks = [f"<loc{round(c * 1023):04d}>" for c in box_]
    names = ["y_min", "x_min", "y_max", "x_max"]
    label(ax, C, 2.65, 2.25, "normalise to [0, 1], ×1023, round:", fs=9, ha="left", color=C.ink)
    for k, (n_, c_, t_) in enumerate(zip(names, box_, toks)):
        label(ax, C, 2.75 + k * 1.25, 1.85, f"{n_} = {c_:.3f}", fs=8.5, color=C.ink, ha="left")
    xs = chips(ax, C, 2.75, 1.1, toks + ["red", "toolbox"], ["orange"] * 4 + ["blue", "blue"], h=0.32, fs=8, mono=True)
    label(ax, C, 2.75, 0.65, "the model writes these 6 tokens one at a time, trained with ordinary cross-entropy",
          fs=8.5, ha="left")
    label(ax, C, 2.75, 0.35, "same mechanism as action tokens: a continuous value → one of K bins → one token",
          fs=8.5, ha="left", color=C["orange"])
    return fig


@figure
def modality_gap(C):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=C.faint, lw=1.2))
    rng = np.random.default_rng(1)
    ci, ct = math.radians(50), math.radians(130)
    for c_, hue, name, spread in [(ci, "blue", "images", 0.1), (ct, "orange", "captions", 0.16)]:
        a_ = c_ + rng.normal(0, spread, 60)
        ax.plot(np.cos(a_), np.sin(a_), "o", color=C[hue], ms=4, alpha=0.8)
        ax.text(1.22 * math.cos(c_), 1.22 * math.sin(c_), name, color=C[hue], ha="center", fontsize=10)
    ax.annotate("", xy=(0.97 * math.cos(ct), 0.97 * math.sin(ct)), xytext=(0.97 * math.cos(ci), 0.97 * math.sin(ci)),
                arrowprops=dict(arrowstyle="<->", color=C.ink, lw=1.2, connectionstyle="arc3,rad=0.3"))
    ax.text(0, 0.62, "the gap", ha="center", fontsize=9, color=C.ink)
    ax.text(0, -0.25, "CLIP ViT-B/32 on depot scenes (Lab 07)\nmatched image–caption cos 0.30\n"
            "image–image 0.87, caption–caption 0.70", ha="center", fontsize=8.5, color=C.ink)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.1, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


@figure
def bag_of_words(C):
    fig, ax = canvas(7.8, 3.0)
    import random as _r
    from vlacourse.data import scenes as S_
    s1 = S_.Scene((S_.Obj("toolbox", "red", 0.3, 0.5), S_.Obj("crate", "blue", 0.72, 0.5)))
    s2 = S_.Scene((S_.Obj("toolbox", "blue", 0.3, 0.5), S_.Obj("crate", "red", 0.72, 0.5)))
    for k, s_ in enumerate([s1, s2]):
        ax.imshow(S_.render(s_, 96, size=0.3, texture=0), extent=(0.2, 1.4, 1.65 - k * 1.45, 2.85 - k * 1.45))
    caps = ["'a red toolbox left of a blue crate'", "'a blue toolbox left of a red crate'"]
    for k, c_ in enumerate(caps):
        box(ax, C, 4.9, 2.15 - k * 1.45, 2.8, 0.4, c_, "orange", fs=8)
    for k in range(2):
        for j in range(2):
            ax.plot([1.45, 4.85], [2.25 - k * 1.45, 2.35 - j * 1.45], color=C.muted if k != j else C["blue"],
                    lw=1.0 if k != j else 1.8, ls="-" if k == j else (0, (3, 2)))
    label(ax, C, 3.1, 2.6, "cos ≈ 0.29", fs=8.5, color=C["blue"])
    label(ax, C, 3.1, 1.45, "cos ≈ 0.29 for all four pairs", fs=8.5, color=C.ink)
    label(ax, C, 3.1, 0.35, "same words, different meaning → nearly the same text embedding;\n"
          "the loss never required telling them apart", fs=8.5, color=C.ink)
    return fig


# ======================================================================= Chapter 5

@figure
def vlm_sequence(C):
    fig, ax = canvas(8, 2.9)
    seq = [("USER:", "faint"), ("img", "blue"), ("img", "blue"), ("…", "blue"), ("img", "blue"), ("What", "aqua"),
           ("is", "aqua"), ("the", "aqua"), ("robot", "aqua"), ("holding?", "aqua"), ("ASSISTANT:", "faint"),
           ("A", "orange"), ("red", "orange"), ("toolbox.", "orange"), ("</s>", "orange")]
    x, xs = 0.1, []
    for t, k in seq:
        w = max(0.36, chip_w(t, 7.5) + 0.06)
        hue = None if k == "faint" else k
        box(ax, C, x, 1.55, w, 0.34, t, hue, fs=7.5, fill=k != "faint", family="monospace")
        xs.append((x, w, k))
        x += w + 0.05
    for (x0, w, k) in xs:
        on = k == "orange"
        label(ax, C, x0 + w / 2, 1.2, "1" if on else "0", fs=8.5, color=C["orange"] if on else C.muted)
    label(ax, C, 0.1, 0.9, "loss mask: only the answer tokens are trained; everything is read", fs=8.5, ha="left")
    bx = xs[1][0]
    ex = xs[4][0] + xs[4][1]
    ax.plot([bx, bx, ex, ex], [2.0, 2.1, 2.1, 2.0], color=C["blue"], lw=1)
    label(ax, C, (bx + ex) / 2, 2.32, "$N_v$ image tokens from the connector\n(576 in LLaVA-1.5, 256 in PaliGemma, 64 in SmolVLM)",
          fs=8, color=C["blue"])
    qx0, qx1 = xs[5][0], xs[9][0] + xs[9][1]
    ax.plot([qx0, qx0, qx1, qx1], [2.0, 2.1, 2.1, 2.0], color=C["aqua"], lw=1)
    label(ax, C, (qx0 + qx1) / 2, 2.32, "question $x$", fs=8, color=C["aqua"])
    ax0, ax1 = xs[11][0], xs[14][0] + xs[14][1]
    ax.plot([ax0, ax0, ax1, ax1], [2.0, 2.1, 2.1, 2.0], color=C["orange"], lw=1)
    label(ax, C, (ax0 + ax1) / 2, 2.32, "answer $y$", fs=8, color=C["orange"])
    label(ax, C, 0.1, 0.45, "one causal language model over the whole sequence; prefill covers everything up to "
          "ASSISTANT:, then one decode pass per answer token", fs=8.5, ha="left", color=C.ink)
    return fig


@figure
def training_stages(C):
    fig, ax = canvas(8, 3.4)
    cols = ["vision encoder", "connector", "language model"]
    stages = [("stage 0: pretrain separately", ["trained", "—", "trained"], "web images / image–text pairs; web text"),
              ("stage 1: alignment", ["frozen", "trained", "frozen"], "image–caption pairs (LLaVA: ~0.6 M)"),
              ("stage 2: visual instruction tuning", ["frozen", "trained", "trained"], "instruction / VQA data (LLaVA-1.5: ~0.7 M)"),
              ("(VLA) stage 3: robot data", ["trained, low LR", "trained", "trained"], "demonstrations + co-training (Ch. 9)")]
    for j, c_ in enumerate(cols):
        label(ax, C, 3.15 + j * 1.25, 3.2, c_, fs=8.5, color=C.ink, weight="bold")
    label(ax, C, 7.05, 3.2, "data", fs=8.5, color=C.ink, weight="bold")
    for i, (name, st, data) in enumerate(stages):
        y = 2.55 - i * 0.68
        label(ax, C, 0.05, y + 0.18, name, fs=8.5, ha="left", color=C.ink)
        for j, s_ in enumerate(st):
            hue = {"trained": "orange", "frozen": "blue", "—": None}.get(s_, "orange")
            txt = {"trained": "🔥 trained", "frozen": "❄ frozen"}.get(s_, s_)
            box(ax, C, 2.6 + j * 1.25, y, 1.1, 0.36, txt.replace("🔥 ", "").replace("❄ ", ""), hue, fs=8,
                fill=s_ != "—", ls="-" if s_ != "—" else (0, (2, 2)))
        label(ax, C, 6.45, y + 0.18, data, fs=7.5, ha="left", color=C.muted)
    return fig


@figure
def smolvlm_tokens(C):
    fig, ax = canvas(8, 3.3)
    img = _explicit([("toolbox", "red", 0.3, 0.65), ("crate", "blue", 0.75, 0.3), ("barrel", "yellow", 0.6, 0.75)], 128)
    # path A: one view
    label(ax, C, 0.1, 3.15, "no splitting", fs=9.5, ha="left", color=C.ink, weight="bold")
    ax.imshow(img, extent=(0.1, 1.0, 2.0, 2.9))
    arrow(ax, C, (1.05, 2.45), (1.35, 2.45))
    label(ax, C, 1.95, 2.62, "SigLIP-B/16", fs=8, color=C.ink)
    cells(ax, C, 1.45, 2.4, 1, 8, 0.12, "blue", fills=[[C.soft("blue", 0.3)] * 8])
    label(ax, C, 1.95, 2.13, "32×32 = 1024 patches", fs=7.5, color=C.muted)
    arrow(ax, C, (2.5, 2.45), (2.8, 2.45))
    label(ax, C, 3.4, 2.62, "pixel shuffle r=4", fs=8, color=C.ink)
    cells(ax, C, 2.9, 2.4, 1, 4, 0.28, "violet", fills=[[C.soft("violet", 0.4)] * 4])
    label(ax, C, 3.45, 2.0, "64 tokens × 12 288", fs=7.5, color=C.muted)
    arrow(ax, C, (4.1, 2.45), (4.4, 2.45))
    box(ax, C, 4.45, 2.27, 1.2, 0.36, "64 image tokens", "orange", fs=8)
    label(ax, C, 6.95, 2.45, "answer: 'three objects: a red,\na blue and a yellow rectangle'", fs=8, color=C.ink)
    # path B: splitting
    label(ax, C, 0.1, 1.55, "splitting (default)", fs=9.5, ha="left", color=C.ink, weight="bold")
    ax.imshow(img, extent=(0.1, 1.0, 0.3, 1.2))
    for k in range(5):
        ax.plot([0.1 + k * 0.225] * 2, [0.3, 1.2], color="white", lw=1)
        ax.plot([0.1, 1.0], [0.3 + k * 0.225] * 2, color="white", lw=1)
    label(ax, C, 0.55, 0.12, "upsampled to 2048 px:\n16 tiles + 1 global view", fs=7.5, color=C.muted)
    arrow(ax, C, (1.05, 0.75), (1.35, 0.75))
    for k in range(17):
        box(ax, C, 1.45 + k * 0.17, 0.62, 0.13, 0.26, "", "orange", r=0.02, lw=0.7)
    label(ax, C, 2.9, 0.4, "17 × 64 = 1088 image tokens (17× prefill)", fs=8, color=C.ink)
    label(ax, C, 6.95, 0.75, "answer: 'a grid of squares on a\nlight gray background'", fs=8, color=C.ink)
    return fig


@figure
def autonomy_ladder(C):
    fig, ax = canvas(8, 3.8)
    rungs = [("1  perception", "answers, boxes", "event-driven", "blue"),
             ("2  task planner", "skill names (SayCan)", "seconds", "aqua"),
             ("3  code as policy", "programs over a robot API", "seconds–minutes", "violet"),
             ("4  spatial outputs", "points, waypoints, value maps", "1–10 Hz", "yellow"),
             ("5  VLA", "motor commands $\\mathbf{a}_t$", "3–50 Hz", "orange")]
    for i, (name, out, rate, hue) in enumerate(rungs):
        y = 3.15 - i * 0.62
        box(ax, C, 0.3 + i * 0.25, y, 2.2, 0.45, name, hue, fs=9, weight="bold")
        label(ax, C, 2.75 + i * 0.25, y + 0.22, out, fs=8.5, ha="left", color=C.ink)
        label(ax, C, 6.0, y + 0.22, rate, fs=8.5, ha="left", color=C.muted)
    label(ax, C, 6.0, 3.75, "decision rate", fs=8.5, ha="left", color=C.ink, weight="bold")
    label(ax, C, 2.75, 3.75, "what the model outputs", fs=8.5, ha="left", color=C.ink, weight="bold")
    arrow(ax, C, (0.12, 3.5), (0.12, 0.6), color=C.muted, lw=1.2)
    label(ax, C, 0.06, 2.0, "toward the actuators: more general,\nmore data, more risk", fs=8,
          rotation=90, color=C.muted)
    return fig


# Values measured in Lab 08 (labs/08_vlm/08_vlm.ipynb, seed 0); update them if the lab changes.
LAB08_STEPS = [400, 800, 1200, 1600, 2000]
LAB08_1D = {"count": [0.99, 0.99, 0.977, 0.997, 0.997], "colour": [0.696, 0.898, 0.945, 0.965, 0.971],
            "object": [0.644, 0.87, 0.926, 0.961, 0.974], "relation": [0.557, 0.845, 0.885, 0.96, 0.971],
            "location": [0.194, 0.372, 0.719, 0.819, 0.839]}
LAB08_2D_LOCATION = [0.452, 0.549, 0.766, 0.855, 0.9]
LAB08_HALLUCINATION = {"yellow": 132, "red": 100, "green": 98, "blue": 70}


@figure
def vlm_skill_curves(C):
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for (k, v), hue in zip(LAB08_1D.items(), ["aqua", "blue", "violet", "yellow", "orange"]):
        ax.plot(LAB08_STEPS, v, "-o", color=C[hue], ms=4, lw=1.8, label=k)
    ax.plot(LAB08_STEPS, LAB08_2D_LOCATION, "--o", color=C["orange"], ms=4, lw=1.5, label="location, 2-D positions")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(300, 2100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("instruction-tuning step (stage 2)")
    ax.set_ylabel("exact-match accuracy")
    ax.grid(True, alpha=0.4)
    return fig


@figure
def hallucination(C):
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    names = list(LAB08_HALLUCINATION) + ["'there is none'"]
    vals = list(LAB08_HALLUCINATION.values()) + [0]
    ax.barh(names[::-1], vals[::-1], color=[C.faint] + [C["orange"]] * 4, height=0.6)
    for k, v in enumerate(vals[::-1]):
        ax.text(v + 3, k, str(v), va="center", fontsize=8.5)
    ax.set_xlabel("answers (400 questions about absent objects)")
    ax.set_title("'what color is the cone?' — asked when there is no cone", fontsize=9.5)
    ax.set_xlim(0, 160)
    return fig


# ======================================================================= Chapter 6

# A 1-D bimodal action distribution: mixture of N(-m, s^2) and N(+m, s^2), equal weights. Its noised
# marginals are Gaussian mixtures too, so the exact score and flow velocity are available in closed form.
GM_M, GM_S = 1.0, 0.15


def _gm_score_vp(x, ab):
    """Score of the VP-noised mixture at signal level ab = alpha_bar."""
    var = ab * GM_S ** 2 + 1 - ab
    comps = [(sgn * math.sqrt(ab) * GM_M) for sgn in (-1, 1)]
    logw = np.stack([-(x - mu) ** 2 / (2 * var) for mu in comps])
    w = np.exp(logw - logw.max(0))
    w /= w.sum(0)
    return sum(wk * (-(x - mu) / var) for wk, mu in zip(w, comps))


def _gm_velocity(x, t):
    """Exact marginal velocity E[x1 - x0 | x_t = x] for x_t = (1-t) x0 + t x1, x0 ~ N(0,1)."""
    var = (1 - t) ** 2 + t ** 2 * GM_S ** 2
    out, logw = [], []
    for sgn in (-1, 1):
        m = sgn * GM_M
        logw.append(-(x - t * m) ** 2 / (2 * var))
        out.append(m + (t * GM_S ** 2 - (1 - t)) * (x - t * m) / var)
    logw = np.stack(logw)
    w = np.exp(logw - logw.max(0))
    w /= w.sum(0)
    return w[0] * out[0] + w[1] * out[1]


@figure
def mode_averaging(C):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.add_patch(Rectangle((-0.25, 0.9), 0.5, 0.35, fc=C.faint, ec=C.muted, lw=1))
    ax.text(0, 1.075, "obstacle", ha="center", va="center", fontsize=9, color=C.ink)
    for k in range(14):
        side = 1 if k % 2 else -1
        y = np.linspace(0, 2.2, 50)
        x = side * 0.5 * np.exp(-((y - 1.07) / 0.45) ** 2) * (1 + 0.15 * rng.normal()) + 0.02 * rng.normal()
        ax.plot(x, y, color=C["blue"], lw=1, alpha=0.6)
    y = np.linspace(0, 2.2, 50)
    ax.plot(np.zeros_like(y), y, color=C["red"], lw=2.5)
    ax.text(-0.95, 0.35, "MSE policy:\nthe average", color=C["red"], fontsize=9)
    ax.annotate("", xy=(-0.02, 0.5), xytext=(-0.55, 0.45), arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1))
    ax.plot(0.5 * np.exp(-((y - 1.07) / 0.45) ** 2), y, color=C["orange"], lw=2.5, ls="--")
    ax.text(0.55, 1.5, "a sample from a\ngenerative policy", color=C["orange"], fontsize=9)
    ax.text(-0.9, 1.9, "demonstrations\n(left and right)", color=C["blue"], fontsize=9)
    ax.plot(0, 0, "o", color=C.ink, ms=7)
    ax.text(0.04, -0.08, "start", fontsize=8.5, color=C.ink)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.15, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


@figure
def action_binning(C):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.3))
    K = 16
    centres = (np.arange(K) + 0.5) / K * 2 - 1
    p = 0.5 * np.exp(-((centres + 0.55) / 0.12) ** 2) + 0.5 * np.exp(-((centres - 0.55) / 0.12) ** 2)
    p /= p.sum()
    a.bar(centres, p, width=2 / K * 0.85, color=C["blue"])
    a.set_title("dimension 1 (lateral): $p(k_1\\mid\\mathbf{o})$, K = 16 bins", fontsize=9)
    a.set_xlabel("action value (bin centres)")
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.annotate("sampled: bin 3", xy=(centres[3], p[3]), xytext=(-0.25, p.max() * 0.95), fontsize=8.5,
               color=C["orange"], arrowprops=dict(arrowstyle="-|>", color=C["orange"]))
    q = np.exp(-((centres - 0.8) / 0.1) ** 2)
    q /= q.sum()
    b.bar(centres, q, width=2 / K * 0.85, color=C["violet"])
    b.set_title("dimension 2 given dimension 1:\n$p(k_2\\mid\\mathbf{o},k_1=3)$ — follows the chosen mode", fontsize=9)
    b.set_xlabel("action value")
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    return fig


@figure
def cvae(C):
    fig, ax = canvas(8, 3.4)
    label(ax, C, 0.1, 3.2, "training", fs=9.5, ha="left", color=C.ink, weight="bold")
    box(ax, C, 0.1, 2.25, 1.3, 0.5, "demonstrated\naction chunk $\\mathbf{a}$", "orange", fs=8)
    box(ax, C, 0.1, 1.55, 1.3, 0.5, "observation $\\mathbf{o}$", "blue", fs=8.5)
    box(ax, C, 1.8, 1.9, 1.3, 0.5, "encoder\n$q_\\phi(\\mathbf{z}\\mid\\mathbf{a},\\mathbf{o})$", "violet", fs=8)
    arrow(ax, C, (1.4, 2.5), (1.8, 2.25))
    arrow(ax, C, (1.4, 1.8), (1.8, 2.05))
    box(ax, C, 3.5, 1.9, 1.5, 0.5, "$\\mathbf{z}=\\boldsymbol{\\mu}+\\boldsymbol{\\sigma}\\odot\\boldsymbol{\\epsilon}$", "violet", fs=9)
    arrow(ax, C, (3.1, 2.15), (3.5, 2.15))
    box(ax, C, 5.4, 1.9, 1.3, 0.5, "decoder\n$D_\\theta(\\mathbf{o},\\mathbf{z})$", "aqua", fs=8.5)
    arrow(ax, C, (5.0, 2.15), (5.4, 2.15))
    ax.annotate("", xy=(5.6, 1.9), xytext=(1.4, 1.65), arrowprops=dict(arrowstyle="-|>", color=C.muted, lw=1,
                                                                       connectionstyle="arc3,rad=0.12"))
    label(ax, C, 7.1, 2.4, "reconstruct $\\mathbf{a}$\n(MSE / L1)", fs=8.5, color=C.ink)
    arrow(ax, C, (6.7, 2.15), (6.95, 2.15))
    label(ax, C, 4.25, 2.7, "+ β · KL(q ‖ N(0, I))", fs=8.5, color=C["violet"])
    # test
    ax.plot([0.1, 7.9], [1.25, 1.25], color=C.faint, lw=1)
    label(ax, C, 0.1, 1.05, "test", fs=9.5, ha="left", color=C.ink, weight="bold")
    box(ax, C, 0.1, 0.25, 1.3, 0.5, "observation $\\mathbf{o}$", "blue", fs=8.5)
    box(ax, C, 3.5, 0.25, 1.5, 0.5, "$\\mathbf{z}\\sim\\mathcal{N}(0,I)$\nor $\\mathbf{z}=\\mathbf{0}$ (ACT)", "violet", fs=8)
    box(ax, C, 5.4, 0.25, 1.3, 0.5, "decoder", "aqua", fs=8.5)
    arrow(ax, C, (5.0, 0.5), (5.4, 0.5))
    ax.annotate("", xy=(5.6, 0.25), xytext=(1.4, 0.4), arrowprops=dict(arrowstyle="-|>", color=C.muted, lw=1,
                                                                      connectionstyle="arc3,rad=0.1"))
    arrow(ax, C, (6.7, 0.5), (6.95, 0.5))
    label(ax, C, 7.35, 0.5, "action\nchunk", fs=8.5, color=C.ink)
    label(ax, C, 2.4, 0.95, "encoder discarded", fs=8, color=C.muted)
    return fig


@figure
def diffusion_paths(C):
    K = 100
    betas = np.linspace(1e-4, 0.05, K)
    ab = np.cumprod(1 - betas)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.25))
    xs = np.linspace(-3, 3, 400)
    for k, hue in [(0, "blue"), (25, "aqua"), (50, "violet"), (99, "orange")]:
        var = ab[k] * GM_S ** 2 + 1 - ab[k]
        dens = sum(0.5 * np.exp(-(xs - s_ * math.sqrt(ab[k]) * GM_M) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
                   for s_ in (-1, 1))
        a.plot(xs, dens, color=C[hue], lw=1.6, label=f"k = {k + 1}")
    a.set_title("forward: noise destroys the modes", fontsize=9.5)
    a.set_yticks([])
    a.spines["left"].set_visible(False)
    a.legend(fontsize=7.5)
    a.set_xlabel("action")
    rng = np.random.default_rng(2)
    x = rng.normal(size=24)
    traj = [x.copy()]
    for k in reversed(range(K)):          # DDPM ancestral sampling with the exact score
        eps = -math.sqrt(1 - ab[k]) * _gm_score_vp(x, ab[k])
        x = (x - betas[k] / math.sqrt(1 - ab[k]) * eps) / math.sqrt(1 - betas[k])
        if k > 0:
            x = x + math.sqrt(betas[k]) * rng.normal(size=x.shape)
        traj.append(x.copy())
    T_ = np.array(traj)
    for j in range(T_.shape[1]):
        b.plot(np.arange(K + 1), T_[:, j], color=C["blue"] if T_[-1, j] < 0 else C["orange"], lw=0.9, alpha=0.8)
    b.axhline(GM_M, color=C.faint, ls="--", lw=1)
    b.axhline(-GM_M, color=C.faint, ls="--", lw=1)
    b.set_title("reverse: stochastic denoising, every path ends in a mode", fontsize=9.5)
    b.set_xlabel("denoising step (noise → data)")
    b.set_ylabel("action")
    return fig


@figure
def flow_paths(C):
    N = 100
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=24)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    # DDIM (probability-flow) with the exact score of a VP process (linear beta schedule in continuous time)
    K = N
    betas = np.linspace(1e-4, 0.05, K)
    ab = np.cumprod(1 - betas)
    x = x0.copy()
    traj = [x.copy()]
    for k in reversed(range(1, K)):
        eps = -math.sqrt(1 - ab[k]) * _gm_score_vp(x, ab[k])
        x0hat = (x - math.sqrt(1 - ab[k]) * eps) / math.sqrt(ab[k])
        x = math.sqrt(ab[k - 1]) * x0hat + math.sqrt(1 - ab[k - 1]) * eps
        traj.append(x.copy())
    T1 = np.array(traj)
    taus = np.linspace(0, 1, len(T1))
    for j in range(T1.shape[1]):
        a.plot(taus, T1[:, j], color=C["blue"] if T1[-1, j] < 0 else C["orange"], lw=0.9)
    a.set_title("diffusion, probability-flow ODE (DDIM)", fontsize=9.5)
    x = x0.copy()
    traj = [x.copy()]
    for i in range(N):
        x = x + _gm_velocity(x, i / N) / N
        traj.append(x.copy())
    T2 = np.array(traj)
    for j in range(T2.shape[1]):
        b.plot(np.linspace(0, 1, N + 1), T2[:, j], color=C["blue"] if T2[-1, j] < 0 else C["orange"], lw=0.9)
    b.set_title("flow matching, linear paths", fontsize=9.5)
    for ax in (a, b):
        ax.set_xlabel("flow time τ (0 = noise, 1 = data)")
        ax.axhline(GM_M, color=C.faint, ls="--", lw=1)
        ax.axhline(-GM_M, color=C.faint, ls="--", lw=1)
    a.set_ylabel("action")
    return fig


@figure
def chunk_timeline(C):
    fig, ax = canvas(8, 2.8)
    infer, exe = 0.9, 1.6
    label(ax, C, 0.05, 2.45, "synchronous", fs=9, ha="left", color=C.ink, weight="bold")
    x = 0.2
    for k in range(3):
        box(ax, C, x, 1.85, infer, 0.35, "infer", "violet", fs=8)
        box(ax, C, x + infer + 0.03, 1.85, exe, 0.35, f"execute chunk {k + 1}", "orange", fs=8)
        x += infer + exe + 0.06
    label(ax, C, 0.2 + infer / 2, 1.65, "robot waits", fs=7.5, color=C["violet"])
    label(ax, C, 0.05, 1.3, "asynchronous", fs=9, ha="left", color=C.ink, weight="bold")
    box(ax, C, 0.2, 0.7, infer, 0.35, "infer 1", "violet", fs=8)
    x = 0.2 + infer + 0.03
    for k in range(4):
        box(ax, C, x, 0.7, exe, 0.35, f"execute chunk {k + 1}", "orange", fs=8)
        box(ax, C, x + exe - infer - 0.05, 0.25, infer, 0.3, f"infer {k + 2}", "violet", fs=7.5)
        x += exe + 0.03
    label(ax, C, 3.6, 0.08, "next chunk computed during execution, from an observation $t_\\mathrm{chunk}$ old",
          fs=8, color=C.muted)
    arrow(ax, C, (0.2, 2.35), (7.9, 2.35), color=C.faint, lw=0.8)
    label(ax, C, 7.9, 2.48, "time", fs=8, ha="right")
    return fig


# ======================================================================= Chapter 7

# Mean path error every 20 steps, measured in Lab 10 (labs/10_imitation_rl, seed 0, 256 episodes).
LAB10_STEPS = list(range(0, 400, 20)) + [399]
LAB10 = {
    "expert": [0.0025, 0.0057, 0.0025, 0.0063, 0.0047, 0.0041, 0.004, 0.0072, 0.0048, 0.0055, 0.0052, 0.0029, 0.0077,
               0.0045, 0.0044, 0.0086, 0.0065, 0.0049, 0.0062, 0.0031, 0.0049],
    "BC, 5 demos": [0.0031, 0.015, 0.0203, 0.0253, 0.0271, 0.0307, 0.0342, 0.0379, 0.0408, 0.0433, 0.0449, 0.0469, 0.0476,
                    0.0501, 0.0517, 0.0521, 0.0533, 0.0561, 0.0572, 0.0586, 0.0595],
    "BC, 50 demos": [0.0031, 0.0154, 0.0208, 0.0261, 0.0277, 0.031, 0.0341, 0.0371, 0.0396, 0.0429, 0.0439, 0.0462, 0.0479,
                     0.05, 0.0527, 0.0529, 0.0539, 0.0564, 0.0581, 0.0604, 0.0615],
    "DAgger, 5 iterations": [0.0032, 0.0064, 0.0057, 0.0062, 0.0063, 0.0064, 0.0058, 0.0062, 0.0057, 0.0058, 0.0062, 0.0058,
                             0.0061, 0.0062, 0.0059, 0.0061, 0.0056, 0.0059, 0.0063, 0.0061, 0.0061],
}


@figure
def pomdp_loop(C):
    fig, ax = canvas(8, 3.0)
    box(ax, C, 0.2, 1.0, 2.0, 1.2, "environment\n(hidden state $\\mathbf{s}_t$:\nposes, contacts, mass)", "faint" if False else "blue",
        fs=8.5)
    box(ax, C, 3.3, 1.75, 1.7, 0.5, "observation $\\mathbf{o}_t$\nimages, $\\mathbf{q}_t$", "aqua", fs=8)
    box(ax, C, 3.3, 0.95, 1.7, 0.5, "instruction $\\ell$", "orange", fs=8.5)
    box(ax, C, 5.6, 1.2, 1.6, 0.8, "policy\n$\\pi_\\theta(\\mathbf{A}_t\\mid\\mathbf{o}_t,\\ell)$", "violet", fs=8.5)
    arrow(ax, C, (2.2, 2.0), (3.3, 2.0))
    label(ax, C, 2.75, 2.15, "$O(\\mathbf{o}\\mid\\mathbf{s})$", fs=8)
    arrow(ax, C, (5.0, 2.0), (5.6, 1.75))
    arrow(ax, C, (5.0, 1.2), (5.6, 1.45))
    ax.annotate("", xy=(1.2, 1.0), xytext=(6.4, 1.2), arrowprops=dict(arrowstyle="-|>", color=C["violet"], lw=1.4,
                                                                     connectionstyle="arc3,rad=-0.3"))
    ax.set_ylim(-0.4, 3.0)
    fig.set_size_inches(8, 3.4)
    label(ax, C, 3.8, -0.22, "action chunk $\\mathbf{A}_t$, executed open loop, changes the state: $P(\\mathbf{s}'\\mid\\mathbf{s},\\mathbf{a})$",
          fs=8.5, color=C["violet"])
    label(ax, C, 6.4, 2.45, "no memory beyond $\\mathbf{o}_t$ (most VLAs)", fs=8, color=C.muted)
    return fig


@figure
def bc_distribution(C):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.linspace(0, 10, 300)
    ref = 0.4 * np.sin(x * 0.6)
    ax.fill_between(x, ref - 0.13, ref + 0.13, color=C.soft("blue", 0.35), lw=0)
    for k in range(6):
        ax.plot(x, ref + 0.05 * np.sin(x * (1 + k * 0.3) + k), color=C["blue"], lw=0.8, alpha=0.7)
    drift = ref + 0.012 * (x - 2).clip(0) ** 1.8
    ax.plot(x, drift, color=C["orange"], lw=2.4)
    ax.text(0.1, -0.75, "expert states $d_{\\pi^\\ast}$ (band): where BC is trained", color=C["blue"], fontsize=9)
    ax.text(6.3, 1.05, "learner states $d_{\\pi_\\theta}$:\nnever trained here", color=C["orange"], fontsize=9)
    ax.plot(2.0, ref[60], "o", color=C["red"], ms=7)
    ax.text(2.15, ref[60] + 0.12, "first small error", color=C["red"], fontsize=8.5)
    ax.set_ylim(-0.9, 1.5)
    ax.set_xlabel("time")
    ax.set_ylabel("state")
    ax.set_xticks([]); ax.set_yticks([])
    return fig


@figure
def compounding_error(C):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for (k, v), hue, ls in zip(LAB10.items(), ["muted", "orange", "yellow", "blue"], ["-", "-", "--", "-"]):
        col = C.muted if hue == "muted" else C[hue]
        ax.plot(LAB10_STEPS, v, ls, color=col, lw=2, label=k)
    ax.set_xlabel("control step")
    ax.set_ylabel("mean distance from the route")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.set_title("patrol task, 256 episodes (Lab 10): training loss is tiny for every policy", fontsize=9)
    return fig


@figure
def dagger_loop(C):
    fig, ax = canvas(7.6, 3.0)
    W, H = 2.0, 0.62
    boxes = {"roll": (0.9, 1.5, "roll out the learner\n(β: sometimes the expert)", "orange"),
             "label": (3.8, 2.45, "expert labels every\nvisited state", "blue"),
             "agg": (6.7, 1.5, "aggregate:\n$\\mathcal{D}\\leftarrow\\mathcal{D}\\cup\\mathcal{D}_i$", "violet"),
             "train": (3.8, 0.45, "retrain by BC\non all of $\\mathcal{D}$", "aqua")}
    for x, y, t, h in boxes.values():
        box(ax, C, x - W / 2, y - H / 2, W, H, t, h, fs=8)
    arrow(ax, C, (1.6, 1.81), (2.8, 2.35))
    arrow(ax, C, (4.8, 2.35), (6.0, 1.81))
    arrow(ax, C, (6.0, 1.19), (4.8, 0.55))
    arrow(ax, C, (2.8, 0.55), (1.6, 1.19))
    label(ax, C, 3.8, 1.48, "the data grows\nwhere the learner goes", fs=8.5, color=C.ink)
    ax.set_xlim(-0.2, 7.8)
    fig.set_size_inches(8.0, 3.0)
    return fig


@figure
def ppo_clip(C):
    rho = np.linspace(0.4, 1.8, 300)
    eps = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=False, gridspec_kw=dict(wspace=0.3))
    for ax, A, t in [(axs[0], 1.0, "advantage > 0"), (axs[1], -1.0, "advantage < 0")]:
        unclipped = rho * A
        obj = np.minimum(rho * A, np.clip(rho, 1 - eps, 1 + eps) * A)
        ax.plot(rho, unclipped, color=C.faint, lw=1.5, ls="--", label="ρ·A (unclipped)")
        ax.plot(rho, obj, color=C["blue"], lw=2.2, label="PPO objective")
        ax.axvline(1 - eps, color=C.muted, lw=0.8, ls=":")
        ax.axvline(1 + eps, color=C.muted, lw=0.8, ls=":")
        ax.set_title(t, fontsize=9.5)
        ax.set_xlabel("probability ratio ρ = π / π_old")
        ax.legend(fontsize=7.5)
    return fig


@figure
def rl_finetune_pipeline(C):
    fig, ax = canvas(8, 2.6)
    stages = [("web pretraining", "VLM: images + text\n(Ch. 3–5)", "blue"), ("imitation", "BC on robot\ndemonstrations (Ch. 7, 9)", "aqua"),
              ("RL fine-tuning", "reward / success signal\n+ KL anchor", "orange")]
    for k, (t, d, h) in enumerate(stages):
        x = 0.2 + k * 2.65
        box(ax, C, x, 1.1, 2.2, 0.9, f"{t}\n{d}", h, fs=8.5)
        if k < 2:
            arrow(ax, C, (x + 2.2, 1.55), (x + 2.6, 1.55))
    label(ax, C, 1.3, 0.7, "knowledge", fs=8.5, color=C["blue"])
    label(ax, C, 3.95, 0.7, "competence: 'often works'", fs=8.5, color=C["aqua"])
    label(ax, C, 6.6, 0.7, "reliability: 'almost always'", fs=8.5, color=C["orange"])
    ax.annotate("", xy=(4.3, 2.0), xytext=(6.6, 2.0), arrowprops=dict(arrowstyle="-|>", color=C["violet"], lw=1.2,
                                                                     connectionstyle="arc3,rad=0.35"))
    label(ax, C, 5.45, 2.45, "KL(π ‖ π_BC): stay close", fs=8.5, color=C["violet"])
    label(ax, C, 4.0, 0.2, "the same three stages as a chat model: pretraining → instruction tuning → RLHF", fs=8.5)
    return fig


# ======================================================================= Chapter 8

@figure
def vla_anatomy(C):
    fig, ax = canvas(8, 3.6)
    img = _explicit([("toolbox", "red", 0.35, 0.6), ("crate", "blue", 0.75, 0.3)], 96, size=0.28)
    ax.imshow(img, extent=(0.1, 0.9, 2.55, 3.35))
    box(ax, C, 1.05, 2.75, 1.05, 0.4, "vision encoder\n+ connector", "blue", fs=7.5)
    box(ax, C, 0.1, 1.85, 2.0, 0.42, "'bring the red toolbox…'\ntokenizer + embedding", "orange", fs=7.5)
    box(ax, C, 0.1, 1.15, 2.0, 0.42, "proprioception $\\mathbf{q}_t$\nlinear projection", "aqua", fs=7.5)
    arrow(ax, C, (0.9, 2.95), (1.05, 2.95))
    xs = 2.35
    tokens = ["blue"] * 6 + ["orange"] * 4 + ["aqua"]
    for k, h in enumerate(tokens):
        box(ax, C, xs, 3.0 - k * 0.2, 0.32, 0.16, "", h, r=0.02, lw=0.7)
    for y0 in (2.95, 2.06, 1.36):
        arrow(ax, C, (2.12, y0), (2.33, y0 if y0 > 2.5 else y0 + 0.2), lw=0.8)
    label(ax, C, xs + 0.16, 0.65, "one token\nsequence", fs=8, color=C.ink)
    box(ax, C, 2.95, 1.1, 1.55, 2.05, "VLM backbone\n$f_\\phi$\n(pretrained)", "violet", fs=9)
    arrow(ax, C, (2.7, 2.1), (2.95, 2.1))
    arrow(ax, C, (4.5, 2.1), (4.85, 2.1))
    label(ax, C, 4.68, 2.3, "$Z_t$", fs=10, color=C.ink)
    heads = [("action tokens (RT-2, OpenVLA)", "orange"), ("flow / diffusion expert (π0, SmolVLA)", "aqua"),
             ("readout + diffusion head (Octo)", "blue"), ("fast System 1 (GR00T N1)", "violet")]
    for k, (t, h) in enumerate(heads):
        box(ax, C, 4.9, 2.95 - k * 0.55, 2.2, 0.42, t, h, fs=7.5)
    arrow(ax, C, (7.15, 2.1), (7.45, 2.1))
    label(ax, C, 7.7, 2.1, "$\\mathbf{A}_t$\n$H_a\\times d_a$", fs=9, color=C.ink)
    label(ax, C, 6.0, 0.55, "action head $h_\\psi$: one of four families", fs=8.5, color=C.ink)
    return fig


@figure
def vla_token_budget(C):
    configs = ["toy VLA\n(this book)", "SmolVLA\n2 cameras", "OpenVLA\n1 camera", "π0\n3 cameras"]
    img = [64, 128, 256, 768]
    txt = [5, 20, 25, 20]
    st = [1, 1, 0, 1]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = np.arange(len(configs))
    ax.barh(y, img, color=C["blue"], height=0.55, label="image tokens")
    ax.barh(y, txt, left=img, color=C["orange"], height=0.55, label="instruction")
    ax.barh(y, st, left=np.array(img) + np.array(txt), color=C["aqua"], height=0.55, label="state")
    for k in range(len(configs)):
        ax.text(img[k] + txt[k] + st[k] + 12, k, f"≈{img[k] + txt[k] + st[k]}", va="center", fontsize=8.5)
    ax.set_yticks(y, configs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("tokens per control step (approximate)")
    ax.set_xlim(0, 900)
    ax.legend(fontsize=8, loc="upper right")
    return fig


@figure
def abs_vs_delta(C):
    rng = np.random.default_rng(0)
    T = 40
    bias = 0.004
    target = np.linspace(0, 1, T)
    absolute = target + bias + 0.003 * rng.normal(size=T)
    delta = np.cumsum(np.diff(np.concatenate([[0], target])) + bias + 0.003 * rng.normal(size=T))
    corr = delta.copy()
    for k in range(0, T, 8):                         # a new observation every 8 steps resets the drift
        corr[k:] -= corr[k] - target[k] if k else 0
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(target, color=C.muted, lw=2, ls=":", label="intended")
    ax.plot(absolute, color=C["blue"], lw=1.8, label="absolute targets: constant error")
    ax.plot(delta, color=C["orange"], lw=1.8, label="delta actions, open loop: error integrates")
    ax.plot(corr, color=C["violet"], lw=1.4, ls="--", label="delta actions, re-observed every 8 steps")
    ax.set_xlabel("control step")
    ax.set_ylabel("position")
    ax.legend(fontsize=7.5)
    return fig


@figure
def fast_dct(C):
    H = 50
    t = np.arange(H)
    traj = 0.6 * np.sin(2 * np.pi * t / 70) + 0.25 * np.tanh((t - 25) / 6) + 0.02 * np.random.default_rng(1).normal(size=H)
    n = np.arange(H)
    M = np.cos(np.pi / H * (n[None, :] + 0.5) * n[:, None])       # DCT-II basis, (k, h)
    coef = M @ traj
    scale = np.where(n == 0, 1 / H, 2 / H)
    keep = coef.copy()
    keep[8:] = 0
    recon = (scale[:, None] * M * keep[:, None]).sum(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 2.8), gridspec_kw=dict(wspace=0.28))
    a.plot(t, traj, color=C["blue"], lw=1.8, label="action dimension over a 50-step chunk")
    a.plot(t, recon, color=C["orange"], lw=1.6, ls="--", label="from 8 of 50 DCT coefficients")
    a.legend(fontsize=7.5, loc="lower center")
    a.set_ylim(-0.55, 0.65)
    a.set_xlabel("step in chunk")
    b.bar(n, np.abs(coef), color=[C["orange"] if k < 8 else C.faint for k in n], width=0.8)
    b.set_xlabel("DCT frequency k")
    b.set_title("|coefficient|: energy in a few low frequencies", fontsize=9)
    return fig


@figure
def expert_coupling(C):
    fig, ax = canvas(8, 3.4)
    titles = ["shared attention (π0)", "cross-attention (GR00T, SmolVLA, toy)", "readout tokens (Octo)"]
    for p, t in enumerate(titles):
        x0 = 0.15 + p * 2.65
        label(ax, C, x0 + 1.15, 3.2, t, fs=9, color=C.ink, weight="bold")
    # 1 shared attention: two weight stacks, layer-wise arrows
    x0 = 0.25
    for l in range(4):
        y = 0.6 + l * 0.55
        box(ax, C, x0, y, 1.1, 0.4, "VLM layer" if l < 3 else "VLM layer", "violet", fs=7)
        box(ax, C, x0 + 1.35, y, 0.85, 0.4, "expert layer", "aqua", fs=7)
        arrow(ax, C, (x0 + 1.1, y + 0.2), (x0 + 1.35, y + 0.2), lw=0.9)
    label(ax, C, x0 + 1.1, 0.35, "actions read the prefix's\nkeys/values in every layer", fs=7.5)
    # 2 cross attention: backbone stack then expert reading final features
    x0 = 2.9
    box(ax, C, x0, 0.6, 1.1, 2.05, "VLM\nbackbone", "violet", fs=8.5)
    for l in range(3):
        y = 0.7 + l * 0.65
        box(ax, C, x0 + 1.35, y, 0.85, 0.45, "self + cross\nattention", "aqua", fs=7)
        ax.annotate("", xy=(x0 + 1.35, y + 0.22), xytext=(x0 + 1.1, 2.4), arrowprops=dict(arrowstyle="-|>", color=C.muted, lw=0.8))
    label(ax, C, x0 + 1.15, 0.35, "expert reads the backbone's\noutput features", fs=7.5)
    # 3 readout
    x0 = 5.55
    box(ax, C, x0, 0.6, 1.3, 2.05, "transformer\n\nobservation tokens\n+ readout tokens", "violet", fs=7.5)
    box(ax, C, x0 + 1.55, 1.35, 0.75, 0.5, "small\ndiffusion\nhead", "blue", fs=7)
    arrow(ax, C, (x0 + 1.3, 1.6), (x0 + 1.55, 1.6))
    label(ax, C, x0 + 1.15, 0.35, "everything passes through\na few readout vectors", fs=7.5)
    return fig


@figure
def chunk_schemes(C):
    fig, ax = canvas(8, 3.3)
    H, steps = 6, 16
    s = 0.42
    rows = [("open-loop chunks, $H_e=4$", "blue"), ("receding horizon, $H_e=1$", "orange"), ("temporal ensembling", "violet")]
    for r, (name, hue) in enumerate(rows):
        y = 2.65 - r * 1.0
        label(ax, C, 0.05, y + 0.32, name, fs=8.5, ha="left", color=C.ink)
        for t in range(steps):
            ax.add_patch(Rectangle((1.3 + t * s, y), s * 0.9, 0.18, fc=C.soft(hue, 0.25), ec=C[hue], lw=0.6))
        if r == 0:
            for q in range(0, steps, 4):
                for h in range(H):
                    if q + h < steps:
                        ax.add_patch(Rectangle((1.3 + (q + h) * s, y + 0.22 + 0.04 * (q // 4 % 2)), s * 0.9, 0.04,
                                               fc=C[hue] if h < 4 else C.faint, ec="none"))
                ax.plot(1.3 + q * s, y - 0.05, "v", color=C.ink, ms=4)
        elif r == 1:
            for q in range(steps):
                ax.plot(1.3 + q * s + 0.2, y - 0.05, "v", color=C.ink, ms=3)
                ax.add_patch(Rectangle((1.3 + q * s, y + 0.22), s * 0.9, 0.05, fc=C[hue], ec="none"))
        else:
            for q in range(steps):
                ax.plot(1.3 + q * s + 0.2, y - 0.05, "v", color=C.ink, ms=3)
                n_cover = min(q + 1, H)
                for k in range(n_cover):
                    w = math.exp(-0.5 * k)
                    ax.add_patch(Rectangle((1.3 + q * s, y + 0.22 + k * 0.06), s * 0.9, 0.05,
                                           fc=C.soft(hue, 0.25 + 0.7 * w), ec="none"))
    label(ax, C, 4.6, 0.25, "▼ = policy query; bars above each step = predictions covering it", fs=8.5)
    return fig


@figure
def latency_budget(C):
    parts = [("camera + transport", 15, "faint"), ("preprocess", 6, "yellow"), ("vision encoder", 12, "blue"),
             ("backbone prefill", 45, "violet"), ("expert × 10", 30, "aqua"), ("post-process + hand-off", 5, "orange")]
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    x = 0
    for name, ms, hue in parts:
        col = C.faint if hue == "faint" else C[hue]
        ax.barh(0, ms, left=x, color=col, height=0.5, edgecolor=C.surface, lw=1.5)
        ax.text(x + ms / 2, 0, name, ha="center", va="center", fontsize=7.5, color=C.on(col), rotation=0 if ms > 20 else 90)
        x += ms
    ax.axvline(160, color=C["red"], ls="--", lw=1.2)
    ax.text(161, 0.33, "deadline: $H_e\\,\\Delta t$ = 8 × 20 ms", fontsize=8, color=C["red"])
    ax.set_yticks([])
    ax.set_xlim(0, 230)
    ax.set_xlabel("milliseconds (illustrative)")
    for s_ in ("left",):
        ax.spines[s_].set_visible(False)
    return fig


# ======================================================================= Chapter 9

@figure
def training_pipeline(C):
    fig, ax = canvas(8, 3.3)
    stages = [("web text", "trillions of tokens", "LM pretraining", "blue", 1.7),
              ("image–text pairs", "billions", "VLM training", "aqua", 1.4),
              ("multi-robot episodes", "millions of steps", "VLA pretraining", "violet", 1.1),
              ("target-robot demos", "hundreds", "post-training", "orange", 0.8)]
    xs = [0.2, 2.2, 4.2, 6.2]
    yc = 2.15
    for x, (data, vol, stage, hue, h) in zip(xs, stages):
        box(ax, C, x, yc - h / 2, 1.6, h, f"{data}\n{vol}", hue, fs=8)
        label(ax, C, x + 0.8, yc + h / 2 + 0.15, stage, fs=8.5, color=C.ink, weight="bold")
    for k in range(3):
        arrow(ax, C, (xs[k] + 1.6, yc), (xs[k + 1], yc))
    for src, dst, rad in [(0, 2, 0.25), (1, 3, 0.25), (2, 3, 0.35)]:
        h_src, h_dst = stages[src][4], stages[dst][4]
        ax.annotate("", xy=(xs[dst] + 0.8, yc - h_dst / 2), xytext=(xs[src] + 0.8, yc - h_src / 2),
                    arrowprops=dict(arrowstyle="-|>", color=C.muted, lw=1.1, ls="--", mutation_scale=10,
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    label(ax, C, 4.0, 0.75, "dashed: co-training keeps samples of earlier data in later stages", fs=8.5)
    label(ax, C, 4.0, 3.2, "data shrinks and specialises  →", fs=8.5)
    ax.set_ylim(0.55, 3.3)
    fig.set_size_inches(8, 2.75)
    return fig


@figure
def teleop_pipeline(C):
    fig, ax = canvas(8.2, 3.0)
    top = [(0.1, 1.0, "operator", "faint"), (1.5, 1.4, "leader device\n(arm, VR, spacemouse)", "orange"),
           (3.4, 1.3, "follower\ncontroller", "violet"), (5.2, 1.3, "follower\nrobot", "blue")]
    y = 2.1
    for x, w, t, h in top:
        box(ax, C, x, y, w, 0.6, t, h if h != "faint" else None, fs=8,
            color=mix(C.faint, C.surface, 0.5) if h == "faint" else None)
    arrow(ax, C, (1.1, y + 0.3), (1.5, y + 0.3))
    arrow(ax, C, (2.9, y + 0.3), (3.4, y + 0.3))
    label(ax, C, 3.15, y + 0.75, "commanded\ntargets", fs=7.5)
    arrow(ax, C, (4.7, y + 0.3), (5.2, y + 0.3))
    box(ax, C, 4.6, 0.55, 1.9, 0.6, "cameras + joint sensors\n(one clock)", "aqua", fs=8)
    arrow(ax, C, (5.85, y), (5.6, 1.15))
    box(ax, C, 6.95, 0.55, 1.15, 0.6, "recorder", "yellow", fs=8.5)
    arrow(ax, C, (6.5, 0.85), (6.95, 0.85))
    ax.annotate("", xy=(7.3, 1.15), xytext=(2.2, y), arrowprops=dict(
        arrowstyle="-|>", color=C["orange"], lw=1.2, ls="--", mutation_scale=10, connectionstyle="arc3,rad=0.12"))
    label(ax, C, 3.0, 1.15, "commanded action\n(the label to imitate)", fs=7.5, color=C["orange"])
    box(ax, C, 6.55, 2.0, 1.55, 0.85, "episode:\nobservations · actions\ncalibration · task\noutcome label", "yellow",
        fs=7, fill=False, ls="--")
    arrow(ax, C, (7.52, 1.15), (7.32, 2.0))
    for (px, py) in [(1.3, y + 0.42), (3.15, y + 0.42), (4.95, y + 0.42), (5.65, 1.55), (6.72, 1.0)]:
        label(ax, C, px, py, "Δt", fs=7, color=C["red"])
    label(ax, C, 4.1, 0.12, "Δt: every arrow has a latency — measure it and log it", fs=8)
    return fig


@figure
def lerobot_format(C):
    fig, ax = canvas(8, 3.4)
    tree = [("meta/", "info: fps, feature schema\nstats: mean / std / min / max\ntasks · episode index", "violet", 2.25),
            ("data/  *.parquet", "one row per frame\nstate · action · timestamps", "blue", 1.2),
            ("videos/  *.mp4", "one stream per camera\ndecoded at the timestamps", "orange", 0.15)]
    for name, desc, hue, y in tree:
        box(ax, C, 0.1, y, 2.5, 0.95, "", hue)
        ax.text(0.22, y + 0.75, name, fontsize=9, family="monospace", weight="bold", color=C.ink, va="center")
        ax.text(0.22, y + 0.33, desc, fontsize=7.5, color=C.ink, va="center")
    cols = ["index", "episode", "timestamp", "state", "action", "task"]
    widths = [0.5, 0.6, 0.75, 0.95, 0.95, 0.45]
    x0, yt, rh = 3.0, 3.2, 0.26
    rows = [["0", "0", "0.000", "[.12, .40]", "[.03, -.01]", "0"], ["1", "0", "0.033", "[.13, .40]", "[.03, -.01]", "0"],
            ["2", "0", "0.067", "[.14, .39]", "[.04, -.02]", "0"], ["3", "0", "0.100", "[.15, .39]", "[.04, -.02]", "0"]]
    x = x0
    for c, w in zip(cols, widths):
        ax.add_patch(Rectangle((x, yt - rh), w, rh, fc=C.soft("blue", 0.45), ec=C["blue"], lw=0.7))
        ax.text(x + w / 2, yt - rh / 2, c, fontsize=7, ha="center", va="center", color=C.ink, family="monospace")
        for r, row in enumerate(rows):
            ax.add_patch(Rectangle((x, yt - (r + 2) * rh), w, rh, fc=C.soft("blue", 0.12), ec=C["blue"], lw=0.5))
        x += w
    x = x0
    for k, w in enumerate(widths):
        for r, row in enumerate(rows):
            ax.text(x + w / 2, yt - (r + 1.5) * rh, row[k], fontsize=6.5, ha="center", va="center", color=C.ink,
                    family="monospace")
        x += w
    arrow(ax, C, (2.6, 1.65), (3.0, 2.3), lw=0.9)
    fy = 0.25
    for k in range(4):
        img = _explicit([("toolbox", "red", 0.55, 0.62), ("crate", "blue", 0.25 + 0.12 * k, 0.35)], 64, size=0.26)
        xf = 3.6 + k * 0.9
        ax.imshow(img, extent=(xf, xf + 0.75, fy, fy + 0.75))
        label(ax, C, xf + 0.375, fy + 0.88, f"t = {rows[k][2]}", fs=7, color=C["orange"])
    label(ax, C, 5.4, 1.4, "video frames decoded at the table's timestamps", fs=7.5, color=C["orange"])
    arrow(ax, C, (2.6, 0.62), (3.55, 0.62), lw=0.9)
    label(ax, C, 5.5, 3.32, "Parquet table (one episode shown)", fs=8)
    return fig


@figure
def lora(C):
    from matplotlib.patches import Polygon
    fig, ax = canvas(7.0, 3.4)
    chips(ax, C, 1.55, 0.15, ["$\\mathbf{x}$"], "faint" if False else "aqua", h=0.3, fs=10, widths=0.5)
    box(ax, C, 0.6, 0.85, 2.4, 1.5, "$W_0$\n$d_\\mathrm{in}\\times d_\\mathrm{out}$\nfrozen", "blue", fs=10)
    ax.add_patch(Polygon([(3.7, 0.85), (5.3, 0.85), (4.75, 1.35), (4.25, 1.35)], closed=True,
                         fc=C.soft("orange"), ec=C["orange"], lw=1.2))
    ax.text(4.5, 1.08, "$A$: $d_\\mathrm{in}\\times r$", ha="center", va="center", fontsize=9)
    ax.add_patch(Polygon([(4.25, 1.85), (4.75, 1.85), (5.3, 2.35), (3.7, 2.35)], closed=True,
                         fc=C.soft("orange"), ec=C["orange"], lw=1.2))
    ax.text(4.5, 2.12, "$B$: $r\\times d_\\mathrm{out}$", ha="center", va="center", fontsize=9)
    label(ax, C, 4.5, 1.6, "rank $r$", fs=8.5, color=C["orange"])
    label(ax, C, 5.95, 1.08, "random init\ntrainable", fs=8, color=C["orange"])
    label(ax, C, 5.95, 2.12, "init 0\ntrainable", fs=8, color=C["orange"])
    arrow(ax, C, (1.8, 0.45), (1.8, 0.85))
    arrow(ax, C, (1.8, 0.6), (4.5, 0.85), rad=0.15)
    ax.add_patch(Circle((3.0, 2.85), 0.17, fc=C.surface, ec=C.muted, lw=1.2))
    ax.text(3.0, 2.85, "+", ha="center", va="center", fontsize=12)
    arrow(ax, C, (1.8, 2.35), (2.83, 2.85))
    arrow(ax, C, (4.5, 2.35), (3.17, 2.85))
    label(ax, C, 4.15, 2.75, "× α / r", fs=8.5, color=C["orange"])
    arrow(ax, C, (3.0, 3.02), (3.0, 3.3))
    label(ax, C, 3.45, 3.25, "$\\mathbf{h}$", fs=10, color=C.ink)
    d, r = 1024, 8
    label(ax, C, 1.8, 0.62 - 0.0, "", fs=8)
    label(ax, C, 5.2, 0.3, f"d = {d}, r = {r}: {2 * d * r // 1000}k trainable vs {d * d / 1e6:.1f}M frozen "
                           f"({2 * d * r / d / d:.1%})", fs=8)
    return fig


def _wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


@figure
def success_ci(C):
    n = np.unique(np.round(np.geomspace(5, 400, 80)).astype(int))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for p, hue in [(0.5, "blue"), (0.8, "orange")]:
        lo, hi = _wilson(p * n, n)
        ax.fill_between(n, lo, hi, color=C.soft(hue, 0.35), lw=0)
        ax.plot(n, lo, color=C[hue], lw=1.2)
        ax.plot(n, hi, color=C[hue], lw=1.2)
        ax.axhline(p, color=C[hue], lw=0.8, ls=":")
        ax.text(420, p, f"measured {p:.0%}", color=C[hue], fontsize=8.5, va="center")
    for nn in (20, 100):
        lo, hi = _wilson(0.5 * nn, nn)
        ax.annotate(f"n = {nn}: [{lo:.2f}, {hi:.2f}]", xy=(nn, lo), xytext=(nn * 1.1, lo - 0.13), fontsize=8,
                    color=C.ink, arrowprops=dict(arrowstyle="-", color=C.muted, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlim(5, 400)
    ax.set_ylim(0.1, 1.05)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 400], ["5", "10", "20", "50", "100", "200", "400"])
    ax.set_xlabel("evaluation episodes n")
    ax.set_ylabel("95% Wilson interval")
    ax.grid(True, alpha=0.4)
    return fig


@figure
def diversity_vs_volume(C):
    k = np.array([1, 2, 4, 8, 16, 32, 64])
    unseen = 0.08 + 0.72 * (1 - np.exp(-k / 14))
    seen = 0.9 - 0.06 * np.log2(k) / 6
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.plot(k, seen, "o-", color=C["blue"], ms=5, label="success where it was trained")
    ax.plot(k, unseen, "o-", color=C["orange"], ms=5, label="success in unseen environments")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k, [str(v) for v in k])
    ax.set_xlabel("environments the fixed demonstration budget is spread over")
    ax.set_ylabel("success (schematic)")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(1, 0.18, "many repetitions,\nfew places", fontsize=8, color=C.muted)
    ax.text(40, 0.42, "few repetitions,\nmany places", fontsize=8, color=C.muted, ha="center")
    return fig


# ======================================================================= Chapter 10

@figure
def vla_timeline(C):
    lanes = [("single-task generative", "aqua", [("Diffusion Policy", 2023.2, 1), ("ACT", 2023.3, -1)]),
             ("multi-task, token actions", "yellow", [("RT-1", 2022.95, 1)]),
             ("VLM → action tokens", "orange", [("RT-2", 2023.55, 1), ("OpenVLA", 2024.45, 1), ("π0-FAST", 2025.04, -1)]),
             ("generalist + action head", "blue", [("Octo", 2024.38, 1), ("OpenVLA-OFT", 2025.15, -1)]),
             ("VLM + action expert", "violet", [("π0", 2024.8, 1), ("GR00T N1", 2025.22, -2), ("π0.5", 2025.3, 1),
                                                ("SmolVLA", 2025.45, -3)])]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for r, (name, hue, items) in enumerate(lanes):
        y = len(lanes) - 1 - r
        ax.axhline(y, color=C.soft(hue, 0.5), lw=6, zorder=0, solid_capstyle="round")
        ax.text(2022.62, y, name, fontsize=8, ha="right", va="center", color=C.ink)
        for t, x, side in items:
            ax.plot(x, y, "o", ms=8, color=C[hue], mec=C.surface, mew=1.5, zorder=3)
            ha = {-2: "right", -3: "left"}.get(side, "center")
            ax.text(x + {-2: 0.02, -3: -0.02}.get(side, 0), y + 0.3 * (1 if side > 0 else -1), t, fontsize=8, ha=ha,
                    va="center", color=C.ink)
    ax.set_xlim(2022.7, 2025.75)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([2023, 2024, 2025], ["2023", "2024", "2025"])
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.5)
    return fig


@figure
def rt1_rt2(C):
    fig, ax = canvas(8, 3.6)
    label(ax, C, 1.9, 3.45, "RT-1 (35 M, robot data only)", fs=9.5, color=C.ink, weight="bold")
    label(ax, C, 6.0, 3.45, "RT-2 (5–55 B, web + robot data)", fs=9.5, color=C.ink, weight="bold")
    img = _explicit([("toolbox", "red", 0.4, 0.55), ("crate", "blue", 0.72, 0.35)], 64, size=0.3)
    for k in range(3):
        ax.imshow(img, extent=(0.1 + k * 0.12, 0.7 + k * 0.12, 2.35 - k * 0.1, 2.95 - k * 0.1), zorder=2 + k)
    label(ax, C, 0.55, 3.12, "6-frame history", fs=7.5)
    box(ax, C, 0.15, 1.35, 1.3, 0.55, "instruction\n(sentence embedding)", "orange", fs=7)
    box(ax, C, 1.2, 2.25, 1.55, 0.7, "EfficientNet\n+ FiLM", "blue", fs=8)
    arrow(ax, C, (0.95, 2.6), (1.2, 2.6))
    arrow(ax, C, (0.8, 1.9), (1.45, 2.25), rad=-0.2)
    box(ax, C, 2.95, 2.25, 0.85, 0.7, "Token\nLearner\n81 → 8", "aqua", fs=7)
    arrow(ax, C, (2.75, 2.6), (2.95, 2.6))
    box(ax, C, 1.9, 1.15, 1.9, 0.6, "transformer (8 layers)", "violet", fs=8)
    arrow(ax, C, (3.37, 2.25), (3.0, 1.75))
    chips(ax, C, 1.55, 0.3, ["k₁", "k₂", "…", "k₁₁"], "orange", h=0.3, fs=8, mono=False)
    arrow(ax, C, (2.85, 1.15), (2.3, 0.62))
    label(ax, C, 3.5, 0.45, "11 dims ×\n256 bins", fs=7.5)
    ax.plot([4.15, 4.15], [0.2, 3.3], color=C.faint, lw=1)
    ax.imshow(img, extent=(4.4, 5.05, 2.3, 2.95))
    box(ax, C, 5.25, 2.3, 0.9, 0.65, "ViT", "blue", fs=8.5)
    arrow(ax, C, (5.05, 2.62), (5.25, 2.62))
    box(ax, C, 6.45, 1.15, 1.45, 1.8, "VLM\nPaLI-X /\nPaLM-E\n(web-pretrained)", "violet", fs=8)
    arrow(ax, C, (6.15, 2.62), (6.45, 2.62))
    box(ax, C, 4.4, 1.2, 1.75, 0.65, "'What should the robot\ndo to pick up the\nextinct animal?'", "orange", fs=6.8)
    arrow(ax, C, (6.15, 1.52), (6.45, 1.52))
    chips(ax, C, 4.45, 0.3, ["1", "128", "91", "241", "5", "…"], "orange", h=0.3, fs=8, mono=True)
    arrow(ax, C, (7.17, 1.15), (6.6, 0.62))
    label(ax, C, 7.45, 0.45, "text tokens\n= action bins", fs=7.5)
    return fig


@figure
def octo_openvla(C):
    fig, ax = canvas(8, 3.5)
    label(ax, C, 1.95, 3.35, "Octo (27–93 M, from scratch)", fs=9.5, color=C.ink, weight="bold")
    label(ax, C, 6.0, 3.35, "OpenVLA (7.5 B, VLM)", fs=9.5, color=C.ink, weight="bold")
    chips(ax, C, 0.15, 2.7, ["T5 text"], "orange", h=0.3, fs=7.5)
    chips(ax, C, 1.1, 2.7, ["image patches"], "blue", h=0.3, fs=7.5)
    chips(ax, C, 2.45, 2.7, ["readouts"], "yellow", h=0.3, fs=7.5, ls="--")
    box(ax, C, 0.15, 1.55, 3.3, 0.8, "transformer, block-wise attention\n(new inputs / heads can be added)", "violet", fs=8)
    for x in (0.55, 1.75, 2.9):
        arrow(ax, C, (x, 2.7), (x, 2.35), lw=0.9)
    box(ax, C, 2.2, 0.55, 1.25, 0.6, "diffusion\nhead", "aqua", fs=8)
    arrow(ax, C, (2.85, 1.55), (2.85, 1.15))
    label(ax, C, 1.2, 0.85, "action chunk", fs=8, color=C.ink)
    arrow(ax, C, (2.2, 0.85), (1.85, 0.85))
    ax.plot([3.85, 3.85], [0.2, 3.2], color=C.faint, lw=1)
    img = _explicit([("toolbox", "red", 0.4, 0.55), ("crate", "blue", 0.72, 0.35)], 64, size=0.3)
    ax.imshow(img, extent=(4.1, 4.65, 2.55, 3.1))
    box(ax, C, 4.9, 2.75, 0.95, 0.35, "SigLIP", "blue", fs=7.5)
    box(ax, C, 4.9, 2.35, 0.95, 0.35, "DINOv2", "aqua", fs=7.5)
    arrow(ax, C, (4.65, 2.83), (4.9, 2.92), lw=0.8)
    arrow(ax, C, (4.65, 2.7), (4.9, 2.52), lw=0.8)
    box(ax, C, 6.05, 2.45, 0.95, 0.55, "concat +\nMLP", "yellow", fs=7.5)
    arrow(ax, C, (5.85, 2.72), (6.05, 2.72))
    box(ax, C, 4.4, 1.35, 3.4, 0.7, "Llama 2 7B (+ prompt: 'What action should the\nrobot take to …?')", "violet", fs=7.5)
    arrow(ax, C, (6.52, 2.45), (6.52, 2.05))
    chips(ax, C, 4.45, 0.75, ["a₁", "a₂", "a₃", "a₄", "a₅", "a₆", "a₇"], "orange", h=0.3, fs=8)
    arrow(ax, C, (5.4, 1.35), (5.4, 1.07))
    label(ax, C, 7.45, 0.9, "7 tokens, one\nat a time (≈6 Hz)", fs=7)
    label(ax, C, 5.9, 0.3, "OFT: parallel decoding · chunks · continuous L1 head", fs=7.5, color=C["orange"])
    return fig


@figure
def pi_series(C):
    fig, ax = canvas(8, 3.6)
    cols = [("π0", 0.1), ("π0-FAST", 2.75), ("π0.5", 5.4)]
    for t, x in cols:
        label(ax, C, x + 1.25, 3.45, t, fs=10, color=C.ink, weight="bold")
        chips(ax, C, x, 2.85, ["images", "instruction"], ["blue", "orange"], h=0.3, fs=7.5)
        box(ax, C, x, 1.95, 1.6 if t == "π0" else 2.45, 0.6, "PaliGemma 3B\n(VLM backbone)", "violet", fs=8)
        arrow(ax, C, (x + 0.6, 2.85), (x + 0.6, 2.55), lw=0.9)
    # pi0: expert beside the backbone
    box(ax, C, 1.85, 1.95, 0.7, 0.6, "expert\n300 M", "aqua", fs=7.5)
    arrow(ax, C, (1.7, 2.25), (1.85, 2.25), lw=0.9)
    label(ax, C, 2.2, 2.85, "noisy chunk\n× 10 steps", fs=7, color=C["aqua"])
    arrow(ax, C, (2.2, 2.68), (2.2, 2.55), lw=0.9)
    box(ax, C, 0.6, 0.9, 1.6, 0.5, "flow-matching\nchunk of 50", "aqua", fs=7.5)
    arrow(ax, C, (2.2, 1.95), (1.6, 1.4), lw=0.9)
    # FAST
    chips(ax, C, 2.85, 1.0, ["t₁", "t₂", "…", "t₃₀"], "orange", h=0.3, fs=7.5)
    arrow(ax, C, (3.9, 1.95), (3.6, 1.3), lw=0.9)
    label(ax, C, 4.0, 0.6, "≈30 FAST tokens (DCT + BPE)\n→ decoded to a 50-step chunk", fs=7)
    # pi0.5
    chips(ax, C, 5.45, 1.25, ["'pick up the plate'"], "yellow", h=0.3, fs=7.5)
    label(ax, C, 7.55, 1.4, "subtask\nin text", fs=7, color=C.muted)
    box(ax, C, 5.45, 0.6, 1.6, 0.42, "flow expert → chunk", "aqua", fs=7.5)
    arrow(ax, C, (6.2, 1.95), (6.2, 1.55), lw=0.9)
    arrow(ax, C, (6.2, 1.25), (6.2, 1.02), lw=0.9)
    label(ax, C, 6.6, 0.25, "co-trained on web data, other robots, subtask labels", fs=7, color=C["violet"])
    for x in (2.6, 5.25):
        ax.plot([x, x], [0.4, 3.3], color=C.faint, lw=1)
    return fig


@figure
def groot_smolvla(C):
    from matplotlib.patches import Polygon
    fig, ax = canvas(8, 3.6)
    label(ax, C, 2.0, 3.45, "GR00T N1: dual system", fs=9.5, color=C.ink, weight="bold")
    label(ax, C, 6.05, 3.45, "SmolVLA: small everywhere", fs=9.5, color=C.ink, weight="bold")
    box(ax, C, 0.1, 2.45, 1.9, 0.7, "System 2: VLM (Eagle-2)\nslow · semantics", "violet", fs=7.5)
    box(ax, C, 0.1, 1.45, 1.9, 0.7, "System 1: DiT flow head\nfast · motor", "aqua", fs=7.5)
    arrow(ax, C, (1.05, 2.45), (1.05, 2.15))
    label(ax, C, 1.5, 2.3, "features", fs=7)
    box(ax, C, 0.1, 0.9, 1.9, 0.32, "embodiment-specific enc / dec", "yellow", fs=7)
    # data pyramid
    px, py, W, Hh = 2.3, 0.35, 1.6, 2.6
    layers = [("human & generated video", "blue"), ("simulation", "aqua"), ("real robot", "orange")]
    for k, (t, hue) in enumerate(layers):
        y0, y1 = py + k * Hh / 3, py + (k + 1) * Hh / 3
        w0, w1 = W * (1 - k / 3), W * (1 - (k + 1) / 3)
        ax.add_patch(Polygon([(px + (W - w0) / 2, y0), (px + (W + w0) / 2, y0), (px + (W + w1) / 2, y1),
                              (px + (W - w1) / 2, y1)], closed=True, fc=C.soft(hue), ec=C[hue], lw=1))
        ax.text(px + W / 2, (y0 + y1) / 2 - (0.05 if k == 2 else 0), t.replace(" & ", " &\n") if k == 0 else t,
                fontsize=6.8 if k < 2 else 6.3, ha="center", va="center", color=C.ink)
    label(ax, C, 3.1, 3.15, "data pyramid", fs=8)
    ax.plot([4.2, 4.2], [0.3, 3.3], color=C.faint, lw=1)
    # SmolVLA
    box(ax, C, 4.45, 2.75, 1.2, 0.4, "64 tokens / cam", "blue", fs=7.5)
    label(ax, C, 6.8, 2.95, "pixel shuffle: 4× fewer image tokens", fs=7)
    for k in range(8):
        on = k < 4
        box(ax, C, 4.45 + k * 0.42, 1.95, 0.36, 0.5, "", "violet" if on else None, fs=7, fill=on, ls="-" if on else ":")
    label(ax, C, 5.15, 1.75, "SmolVLM2 layers: only the first half are used", fs=7, ha="left")
    box(ax, C, 4.45, 0.95, 1.6, 0.5, "expert ≈100 M\ncross ↔ self", "aqua", fs=7.5)
    arrow(ax, C, (5.0, 1.95), (5.0, 1.45), lw=0.9)
    box(ax, C, 6.3, 0.95, 1.6, 0.5, "chunk of 50", "aqua", fs=7.5, fill=False)
    arrow(ax, C, (6.05, 1.2), (6.3, 1.2))
    label(ax, C, 6.15, 0.55, "asynchronous: next chunk computed\nwhile this one executes", fs=7)
    return fig


# ======================================================================= Chapter 11

@figure
def latency_tail(C):
    rng = np.random.default_rng(3)
    n = 20_000
    base = rng.lognormal(np.log(58), 0.12, n)
    spikes = rng.random(n) < 0.03
    lat = np.where(spikes, rng.uniform(110, 260, n), base)
    mean, p99 = lat.mean(), np.percentile(lat, 99)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.hist(lat, bins=np.arange(30, 280, 3), color=C["blue"], edgecolor=C.surface, lw=0.3)
    ax.set_yscale("log")
    for v, name, hue, dy in [(mean, f"mean {mean:.0f} ms", "aqua", 0.0), (p99, f"p99 {p99:.0f} ms", "orange", 0.0)]:
        ax.axvline(v, color=C[hue], lw=1.6, ls="--")
    miss_mean = (lat > 80).mean()
    miss_p99 = (lat > p99).mean()
    ax.text(mean + 4, 2500, f"mean {mean:.0f} ms", color=C.ink, fontsize=8.5)
    ax.text(p99 + 4, 2500, f"p99 {p99:.0f} ms", color=C.ink, fontsize=8.5)
    bb = dict(fc=C.surface, ec="none", pad=1.5)
    ax.text(112, 300, f"budget 80 ms (fits the mean): misses {miss_mean:.1%} of calls\n"
                      f"= {miss_mean * 10 * 60:.0f} misses / min at 10 Hz", fontsize=8, color=C.ink, bbox=bb)
    ax.text(112, 40, f"budget {p99:.0f} ms (fits the p99): misses {miss_p99:.0%}", fontsize=8, color=C.ink, bbox=bb)
    ax.set_xlabel("end-to-end latency (ms), illustrative")
    ax.set_ylabel("calls (log scale)")
    return fig


@figure
def memory_fit(C):
    models = [("SmolVLA 0.45 B", 0.45), ("GR00T N1 ≈2 B", 2.2), ("π0 3.3 B", 3.3), ("OpenVLA 7.5 B", 7.5)]
    fmts = [("bf16", 2.0, "blue"), ("int8", 1.0 + 2 / 64, "aqua"), ("int4", 0.5 + 2 / 64, "orange")]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, (name, p) in enumerate(models):
        for j, (f, b, hue) in enumerate(fmts):
            y = i + (j - 1) * 0.26
            gb = p * b
            ax.barh(y, gb, height=0.22, color=C[hue], label=f if i == 0 else None)
            ax.text(gb + 0.15, y, f"{gb:.1f}", va="center", fontsize=7, color=C.ink)
    for gb, dev, yy in [(6, "6 GB laptop GPU", -0.78), (8, "Orin Nano 8 GB", -0.5), (16, "Orin NX 16 GB", -0.78)]:
        ax.axvline(gb, color=C.muted, lw=1, ls="--")
        ax.text(gb + (0.1 if gb > 6 else -0.1), yy, dev, fontsize=7.5, color=C.muted, ha="left" if gb > 6 else "right")
    ax.set_yticks(range(len(models)), [m for m, _ in models], fontsize=8.5)
    ax.set_ylim(len(models) - 0.5, -0.95)
    ax.set_xlim(0, 21)
    ax.set_xlabel("weight memory (GB); leave room for KV cache, activations and the robot's other software")
    ax.legend(fontsize=8, loc="lower right", title="precision", title_fontsize=8)
    ax.xaxis.label.set_size(8.5)
    return fig


@figure
def policy_server(C):
    fig, ax = canvas(8, 3.3)
    ys = [2.55, 1.8, 1.05, 0.3]
    fill = [3, 6, 1, 5]
    for k, (y, f) in enumerate(zip(ys, fill)):
        box(ax, C, 0.1, y, 1.9, 0.55, "", "blue")
        ax.text(0.2, y + 0.4, f"robot {k + 1}", fontsize=8, color=C.ink, va="center")
        for q in range(8):
            ax.add_patch(Rectangle((0.2 + q * 0.21, y + 0.08), 0.17, 0.16,
                                   fc=C["orange"] if q < f else "none", ec=C["orange"], lw=0.6))
        ax.text(1.95, y + 0.4, "queue", fontsize=6.5, color=C.muted, ha="right", va="center")
        arrow(ax, C, (2.0, y + 0.36), (3.3, 1.75), lw=0.9)
        arrow(ax, C, (3.3, 1.55), (2.0, y + 0.18), lw=0.9, ls="--", color=C["orange"])
    box(ax, C, 3.3, 1.2, 1.5, 0.9, "batcher\nwindow ≤ 5 ms\nmost urgent first", "yellow", fs=7.5)
    arrow(ax, C, (4.8, 1.65), (5.3, 1.65))
    box(ax, C, 5.3, 1.05, 2.5, 1.2, "GPU: one forward pass\nfor the whole batch\n(weights read once)", "violet", fs=8)
    label(ax, C, 2.65, 3.05, "→ obs + $t_\\mathrm{obs}$", fs=7.5, color=C.ink)
    label(ax, C, 2.65, 0.08, "⇠ chunk aligned to $t_\\mathrm{obs}$", fs=7.5, color=C["orange"])
    label(ax, C, 6.55, 0.6, "each robot keeps executing its queue meanwhile", fs=7.5)
    return fig


@figure
def ros2_graph(C):
    fig, ax = canvas(8, 3.7)
    node = lambda x, y, w, h, t, hue="blue", fs=7.5: box(ax, C, x, y, w, h, t, hue, fs=fs, r=0.12)  # noqa: E731
    topic = lambda x, y, t, w=1.45: chips(ax, C, x, y, [t], "yellow", h=0.26, fs=6.8, mono=True, widths=w)  # noqa: E731
    node(0.05, 2.45, 1.4, 0.5, "camera driver")
    node(0.05, 1.4, 1.4, 0.5, "joint state\nbroadcaster")
    topic(1.55, 2.57, "/camera/image_raw")
    topic(1.55, 1.52, "/joint_states")
    node(3.45, 1.3, 1.25, 1.8, "VLA node\n\npolicy +\nchunk\nqueue", "violet", fs=8)
    arrow(ax, C, (1.45, 2.7), (1.55, 2.7), lw=0.8)
    arrow(ax, C, (3.0, 2.7), (3.45, 2.7), lw=0.8)
    arrow(ax, C, (1.45, 1.65), (1.55, 1.65), lw=0.8)
    arrow(ax, C, (3.0, 1.65), (3.45, 1.65), lw=0.8)
    node(3.15, 3.3, 1.85, 0.35, "operator / task planner", "orange", fs=7)
    arrow(ax, C, (4.07, 3.3), (4.07, 3.1), lw=0.9)
    label(ax, C, 5.05, 3.2, "action /vla/execute", fs=6.8, ha="left", family="monospace")
    topic(4.85, 2.07, "/vla/action_chunk")
    arrow(ax, C, (4.7, 2.2), (4.85, 2.2), lw=0.8)
    node(6.55, 1.85, 1.4, 0.7, "safety node\nlimits · workspace\nwatchdog", "red", fs=7)
    arrow(ax, C, (6.3, 2.2), (6.55, 2.2), lw=0.8)
    topic(6.95, 1.25, "/cmd", 0.5)
    arrow(ax, C, (7.25, 1.85), (7.05, 1.51), lw=0.8)
    node(6.55, 0.35, 1.4, 0.55, "ros2_control\ncontroller", "aqua")
    arrow(ax, C, (7.05, 1.25), (7.15, 0.9), lw=0.8)
    node(0.05, 0.25, 1.4, 0.5, "rosbag2\nrecorder", "faint" if False else "aqua")
    ax.plot([1.45, 6.4], [0.5, 0.5], ls=":", color=C.muted, lw=1)
    for x, top_ in ((2.2, 1.52), (4.07, 1.3), (5.6, 2.07)):
        ax.plot([x, x], [0.5, top_], ls=":", color=C.muted, lw=1)
    label(ax, C, 3.9, 0.3, "records every topic (inputs, chunks, filtered commands)", fs=7)
    label(ax, C, 7.25, 3.0, "only this node\ncommands actuators", fs=7, color=C["red"])
    return fig


@figure
def monitoring(C):
    fig, ax = canvas(8, 3.4)
    sig = [("input health: frozen / blurred frames", "blue"), ("OOD score on backbone features", "blue"),
           ("action-sample disagreement", "blue"), ("progress stall", "blue"), ("latency / deadline misses", "blue")]
    for k, (t, hue) in enumerate(sig):
        y = 2.95 - k * 0.48
        box(ax, C, 0.1, y, 2.55, 0.36, t, hue, fs=7.5)
        arrow(ax, C, (2.65, y + 0.18), (3.15, 2.05), lw=0.8)
    box(ax, C, 3.15, 1.6, 1.25, 0.9, "supervisor\n(calibrated\nthresholds)", "violet", fs=8)
    steps = [("log", "aqua"), ("slow down", "yellow"), ("ask for help", "orange"), ("safe stop", "red")]
    for k, (t, hue) in enumerate(steps):
        box(ax, C, 4.75 + k * 0.82, 1.55 + k * 0.38, 0.78, 0.34, t, hue, fs=7.2)
    arrow(ax, C, (4.4, 2.05), (4.75, 1.75))
    label(ax, C, 6.1, 3.2, "escalation", fs=8)
    box(ax, C, 3.15, 0.2, 0.9, 0.45, "VLA", "violet", fs=8)
    box(ax, C, 4.5, 0.2, 1.75, 0.45, "safety filter (always\nauthoritative)", "red", fs=7.2)
    box(ax, C, 6.7, 0.2, 1.2, 0.45, "actuators", None, fs=8, color=mix(C.faint, C.surface, 0.5))
    arrow(ax, C, (4.05, 0.42), (4.5, 0.42))
    arrow(ax, C, (6.25, 0.42), (6.7, 0.42))
    arrow(ax, C, (3.78, 1.6), (5.2, 0.65), lw=0.9, ls="--")
    label(ax, C, 4.95, 1.08, "mode", fs=7)
    return fig


# ======================================================================= Chapter 12

@figure
def sensor_routing(C):
    fig, ax = canvas(8, 3.5)
    sensors = [("joint encoders  1–10 kHz", "aqua"), ("IMU  0.2–8 kHz", "aqua"), ("force-torque  1 kHz", "aqua"),
               ("odometry / VIO  30–200 Hz", "aqua"), ("cameras  15–60 Hz", "blue")]
    for k, (t, hue) in enumerate(sensors):
        box(ax, C, 0.1, 2.95 - k * 0.6, 2.1, 0.42, t, hue, fs=7.5)
    box(ax, C, 3.0, 1.85, 1.55, 0.9, "state estimator\n(Kalman filter,\nVIO)", "violet", fs=7.5)
    box(ax, C, 3.0, 0.05, 1.55, 0.72, "controllers\n0.1–10 kHz", "orange", fs=8)
    box(ax, C, 5.6, 1.6, 1.6, 1.3, "VLA\n1–50 Hz", "yellow", fs=9)
    for k in range(4):
        arrow(ax, C, (2.2, 3.16 - k * 0.6), (3.0, 2.3), lw=0.8)
    for k in (0, 2):
        arrow(ax, C, (2.2, 3.16 - k * 0.6), (3.0, 0.5), lw=0.8, ls=":")
    arrow(ax, C, (2.2, 0.76), (5.6, 1.9), lw=1.2, color=C["blue"])
    label(ax, C, 4.1, 1.53, "images", fs=7.5, color=C["blue"])
    arrow(ax, C, (4.55, 2.3), (5.6, 2.5), lw=1.2, color=C["violet"])
    label(ax, C, 5.0, 2.95, "estimated state\n(compact, normalised)", fs=7, color=C["violet"])
    arrow(ax, C, (3.78, 1.85), (3.78, 0.77), lw=0.9)
    arrow(ax, C, (6.4, 1.6), (4.55, 0.45), lw=1.0, color=C["yellow"], rad=-0.2)
    label(ax, C, 6.55, 0.55, "targets / chunks", fs=7.5)
    return fig


@figure
def control_hierarchy(C):
    fig, ax = canvas(8, 3.7)
    layers = [("task / mission", "operator · planner · VLM", "≤ 1 Hz", "orange"),
              ("skill", "VLA", "1–50 Hz", "yellow"),
              ("safety filter", "limits · geofence · CBF · watchdog", "every cycle", "red"),
              ("motion planning", "trajectories, MPC", "10–100 Hz", "aqua"),
              ("control", "PID · impedance · attitude", "0.1–10 kHz", "blue")]
    for k, (name, what, rate, hue) in enumerate(layers):
        y = 3.05 - k * 0.66
        w = 3.2 + k * 0.6
        x = 4.0 - w / 2
        box(ax, C, x, y, w, 0.5, "", hue)
        ax.text(x + 0.12, y + 0.25, name, fontsize=8.5, weight="bold", va="center", color=C.ink)
        ax.text(x + w - 0.12, y + 0.25, what, fontsize=7.5, va="center", ha="right", color=C.ink)
        label(ax, C, 7.55, y + 0.25, rate, fs=8, color=C.ink)
    arrow(ax, C, (0.15, 3.5), (0.15, 0.45), lw=1.2)
    label(ax, C, 0.32, 1.95, "commands: more concrete, more frequent", fs=7.5, rotation=90)
    label(ax, C, 4.0, 0.12, "rates increase downwards by ≈5–10× per layer; so does the strength of the guarantees", fs=8)
    label(ax, C, 7.55, 3.62, "rate", fs=8, weight="bold")
    return fig


@figure
def learn_vs_classical(C):
    items = [("instruction\nunderstanding", 0.92, 0.12, "orange"), ("object\ngrounding", 0.85, 0.22, "orange"),
             ("grasp choice", 0.7, 0.32, "orange"), ("cloth / cable\nmanipulation", 0.86, 0.42, "orange"),
             ("legged\nlocomotion", 0.6, 0.72, "violet"), ("motion planning", 0.3, 0.6, "aqua"),
             ("state estimation", 0.22, 0.78, "blue"), ("attitude / joint\ncontrol", 0.1, 0.9, "blue"),
             ("geofence,\nlimits", 0.04, 0.98, "red")]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    xx = np.linspace(0, 1, 50)
    ax.fill_between(xx, xx - 0.18, xx + 0.18, color=C.soft("violet", 0.25), lw=0)
    for t, x, y, hue in items:
        ax.plot(x, y, "o", color=C[hue], ms=8)
        ax.text(x + (0.025 if x < 0.75 else -0.025), y, t, fontsize=7.5, va="center",
                ha="left" if x < 0.75 else "right", color=C.ink)
    ax.text(0.97, 0.03, "learn", fontsize=11, ha="right", color=C["orange"], weight="bold")
    ax.text(0.02, 1.02, "keep classical", fontsize=11, ha="left", va="bottom", color=C["blue"], weight="bold")
    ax.text(0.5, 0.36, "research frontier", fontsize=8, color=C["violet"], rotation=29, ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("how hard to model by hand  →")
    ax.set_ylabel("speed / cost of failure  →")
    return fig


@figure
def cbf_filter(C):
    from vlacourse.safety import cbf_filter as filt
    zone_c, zone_r = np.array([0.5, 0.5]), 0.17
    goal = np.array([0.95, 0.62])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.25))
    for start in (np.array([0.05, 0.36]), np.array([0.05, 0.55])):
        x, raw, filt_tr, act = start.copy(), [start.copy()], [start.copy()], []
        y = start.copy()
        go = lambda p: (goal - p) / max(np.linalg.norm(goal - p), 1e-9) * min(1.0, np.linalg.norm(goal - p) / 0.05)  # noqa: E731
        for _ in range(220):
            uf, on = filt(y, go(y), [(zone_c, zone_r)], alpha=4.0)
            y = y + 0.01 * uf
            filt_tr.append(y.copy()); act.append(on)
            x = x + 0.01 * go(x)
            raw.append(x.copy())
        raw, filt_tr, act = np.array(raw), np.array(filt_tr), np.array(act)
        a.plot(raw[:, 0], raw[:, 1], ":", color=C.muted, lw=1.5)
        a.plot(filt_tr[:, 0], filt_tr[:, 1], color=C["blue"], lw=2)
        a.plot(filt_tr[1:][act, 0], filt_tr[1:][act, 1], ".", color=C["orange"], ms=3)
        a.plot(*start, "o", color=C.ink, ms=5)
    a.add_patch(Circle(zone_c, zone_r, fc=C.soft("red", 0.3), ec=C["red"], ls="--"))
    a.plot(*goal, "*", color=C["green"], ms=12)
    a.text(zone_c[0], zone_c[1], "keep-out", ha="center", va="center", fontsize=8, color=C["red"])
    a.plot([], [], ":", color=C.muted, label="policy alone"); a.plot([], [], color=C["blue"], label="filtered")
    a.plot([], [], ".", color=C["orange"], label="filter active")
    a.legend(fontsize=7.5, loc="lower right")
    a.set_xlim(0, 1); a.set_ylim(0.2, 0.85); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    # right: the half-space at one state
    x = np.array([0.27, 0.33])
    u = np.array([1.0, 0.25])
    alpha = 4.0
    av = 2 * (x - zone_c)
    bv = -alpha * (np.sum((x - zone_c) ** 2) - zone_r ** 2)
    uf, _ = filt(x, u, [(zone_c, zone_r)], alpha=alpha, u_max=10)
    g = np.linspace(-1.5, 1.5, 300)
    U, V = np.meshgrid(g, g)
    ok = av[0] * U + av[1] * V >= bv
    b.contourf(U, V, ok, levels=[0.5, 1.5], colors=[C.soft("aqua", 0.35)])
    b.contour(U, V, av[0] * U + av[1] * V - bv, levels=[0], colors=[C["aqua"]], linewidths=1.2)
    b.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C.muted, lw=1.6))
    b.annotate("", xy=uf, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2))
    b.annotate("", xy=uf, xytext=u, arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.2, ls="--"))
    b.text(u[0] + 0.05, u[1] + 0.08, "$\\mathbf{u}_\\mathrm{prop}$", fontsize=10, color=C.ink)
    b.text(uf[0] - 0.05, uf[1] - 0.25, "$\\mathbf{u}^\\ast$", fontsize=10, color=C["blue"])
    b.text(-1.4, 1.25, "allowed: $\\nabla h^\\top\\mathbf{u}\\geq-\\alpha h$", fontsize=8.5, color=C.ink)
    b.set_xlabel("$u_x$"); b.set_ylabel("$u_y$"); b.set_aspect("equal")
    b.axhline(0, color=C.faint, lw=0.6); b.axvline(0, color=C.faint, lw=0.6)
    b.set_title("velocity space at one state", fontsize=9)
    return fig


# ======================================================================= Chapter 13

@figure
def uav_stack(C):
    fig, ax = canvas(8, 3.8)
    box(ax, C, 0.1, 0.25, 3.7, 3.25, "", "blue", fill=False, ls="--")
    label(ax, C, 1.95, 3.32, "flight controller (PX4)", fs=9, color=C.ink, weight="bold")
    loops = [("position / velocity  50–100 Hz", "aqua"), ("attitude  250–500 Hz", "aqua"), ("rate  0.4–8 kHz", "aqua")]
    for k, (t, hue) in enumerate(loops):
        box(ax, C, 0.3, 2.55 - k * 0.55, 2.3, 0.4, t, hue, fs=7.5)
        if k:
            arrow(ax, C, (1.45, 2.55 - (k - 1) * 0.55), (1.45, 2.95 - k * 0.55), lw=0.8)
    box(ax, C, 0.3, 0.45, 1.1, 0.45, "motors", None, fs=8, color=mix(C.faint, C.surface, 0.5))
    arrow(ax, C, (1.0, 1.45), (0.85, 0.9), lw=0.8)
    box(ax, C, 2.75, 1.45, 0.95, 1.5, "EKF2\nestimator", "violet", fs=7.5)
    box(ax, C, 1.55, 0.45, 2.15, 0.45, "failsafes · geofence", "red", fs=7.5)
    box(ax, C, 4.45, 1.7, 2.0, 1.6, "", "yellow", fill=True)
    ax.text(5.45, 3.1, "companion computer", fontsize=8.5, ha="center", weight="bold", color=C.ink)
    for k, t in enumerate(["perception, mapping", "planner + safety filter", "VLA / VLM (0.5–10 Hz)"]):
        ax.text(5.45, 2.75 - k * 0.32, t, fontsize=7.5, ha="center", color=C.ink)
    arrow(ax, C, (4.45, 2.4), (3.8, 2.4))
    arrow(ax, C, (3.8, 2.0), (4.45, 2.0))
    label(ax, C, 4.12, 2.6, "setpoints", fs=7)
    label(ax, C, 4.12, 1.8, "state", fs=7)
    label(ax, C, 4.12, 1.5, "uXRCE-DDS\n/ MAVLink", fs=6.5)
    box(ax, C, 4.45, 0.35, 2.0, 0.9, "cameras, depth,\nlidar", "blue", fs=7.5)
    arrow(ax, C, (5.45, 1.25), (5.45, 1.7))
    box(ax, C, 6.85, 2.35, 1.05, 0.9, "ground\nstation", "orange", fs=7.5)
    box(ax, C, 6.85, 0.95, 1.05, 0.9, "pilot\n(RC)", "red", fs=7.5)
    arrow(ax, C, (6.85, 2.8), (6.45, 2.8), ls="--", lw=0.9)
    ax.annotate("", xy=(3.8, 1.0), xytext=(6.85, 1.4), arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.2,
                                                                       connectionstyle="arc3,rad=0.25"))
    label(ax, C, 6.0, 0.15, "RC override acts directly on the flight controller", fs=7, color=C["red"])
    return fig


@figure
def offboard_timeline(C):
    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    T = 12.0
    hb = np.arange(0, 9.0, 0.1)
    ax.vlines(hb, 2.7, 2.9, color=C["aqua"], lw=0.6)
    ax.text(-0.1, 2.8, "heartbeat +\nsetpoints 10 Hz", ha="right", va="center", fontsize=7.5, color=C.ink)
    dec = [0.5, 1.6, 2.4, 3.5, 4.3, 5.2]
    ax.vlines(dec, 1.9, 2.2, color=C["yellow"], lw=2.5)
    ax.text(-0.1, 2.05, "VLA goals", ha="right", va="center", fontsize=7.5, color=C.ink)
    spans = [(0, 1.0, "warm-up", "faint"), (1.0, 5.6, "OFFBOARD", "blue"), (5.6, 9.0, "watchdog: HOLD", "orange"),
             (9.0, 9.0 + 1.0, "timeout", "faint"), (10.0, T, "PX4 failsafe", "red")]
    for a, b, t, hue in spans:
        col = mix(C.faint, C.surface, 0.5) if hue == "faint" else C.soft(hue, 0.45)
        ax.add_patch(Rectangle((a, 1.0), b - a, 0.5, fc=col, ec=C.muted if hue == "faint" else C[hue], lw=0.8))
        ax.text((a + b) / 2, 1.25, t, ha="center", va="center", fontsize=7 if b - a < 1.5 else 7.5, color=C.ink)
    ax.text(-0.15, 1.25, "vehicle mode", ha="right", va="center", fontsize=7.5, color=C.ink)
    for x, t in [(1.0, "switch to offboard + arm"), (5.6, "VLA silent > 0.5 s"), (9.0, "stream stops (companion fails)"),
                 (10.0, "COM_OF_LOSS_T elapsed")]:
        ax.axvline(x, color=C.muted, ls=":", lw=0.8)
        ax.text(x + 0.05, 0.75 if x != 10.0 else 0.45, t, fontsize=7, color=C.muted, va="center")
    ax.set_xlim(-0.2, T); ax.set_ylim(0.3, 3.1)
    ax.set_yticks([]); ax.set_xlabel("time (s), illustrative")
    for s in ("left",):
        ax.spines[s].set_visible(False)
    return fig


@figure
def uav_architectures(C):
    fig, ax = canvas(8, 3.6)
    cols = [("A  mission writer", ["VLM", "mission\n(waypoints)", "review /\nvalidate", "PX4 mission"]),
            ("B  goal proposer", ["VLA / VLM", "goal", "planner +\nfilter", "PX4 offboard"]),
            ("C  chunk producer", ["VLA", "trajectory\nchunk", "smooth +\nfilter", "PX4 offboard"]),
            ("D  direct velocity", ["VLA", "velocity", "filter", "PX4 offboard"])]
    hues = ["yellow", "violet", "aqua", "blue"]
    for i, (title, steps) in enumerate(cols):
        x0 = 0.15 + i * 1.98
        label(ax, C, x0 + 0.85, 3.42, title, fs=8.5, color=C.ink, weight="bold")
        for k, (t, h) in enumerate(zip(steps, hues)):
            y = 2.75 - k * 0.72
            box(ax, C, x0, y, 1.7, 0.5, t, h, fs=7.5)
            if k:
                arrow(ax, C, (x0 + 0.85, y + 0.72), (x0 + 0.85, y + 0.5), lw=0.9)
    ax.annotate("", xy=(7.9, 0.05), xytext=(0.1, 0.05), arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.4))
    label(ax, C, 5.6, 0.22, "more authority for the model, less for classical checks", fs=8, color=C["red"])
    box(ax, C, 2.08, 0.55, 1.9, 2.8, "", "green", fill=False, lw=2.2)
    label(ax, C, 3.03, 0.4, "recommended", fs=8, color=C["green"], weight="bold")
    return fig


@figure
def delay_oscillation(C):
    k, dt, T = 4.0, 0.002, 4.0
    t = np.arange(0, T, dt)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for delay, hue in [(0.05, "blue"), (0.25, "orange"), (0.45, "red")]:
        nd = int(round(delay / dt))
        x = np.zeros(len(t))
        for i in range(1, len(t)):
            x[i] = x[i - 1] + dt * k * (1.0 - x[max(0, i - 1 - nd)])
        ax.plot(t, x, color=C[hue], lw=1.8, label=f"δ = {delay * 1000:.0f} ms  (kδ = {k * delay:.1f})")
    ax.axhline(1.0, color=C.muted, ls=":", lw=1)
    ax.set_ylim(-0.6, 2.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("stable iff kδ < π/2 ≈ 1.57; no overshoot iff kδ < 1/e ≈ 0.37", fontsize=8.5)
    return fig


@figure
def uav_safety(C):
    fig, ax = canvas(8, 3.9)
    layers = [("VLA / VLM: proposes goals", "yellow", "companion"),
              ("goal validation: fence, no-fly zones, altitude, battery", "violet", "companion"),
              ("planner with margins and limits", "aqua", "companion"),
              ("setpoint filter: saturation, rate limits, CBF", "blue", "companion"),
              ("watchdog: hold when proposals stop", "orange", "companion"),
              ("PX4: geofence, offboard-loss, RC-loss, battery failsafes", "red", "flight controller"),
              ("pilot: RC override, kill switch", "red", "independent")]
    for k, (t, hue, where) in enumerate(layers):
        y = 3.4 - k * 0.48
        w = 4.6 + k * 0.25
        box(ax, C, 0.15, y, w, 0.38, t, hue, fs=7.5)
        label(ax, C, w + 0.35, y + 0.19, where, fs=7, color=C.muted, ha="left")
    ax.annotate("", xy=(7.75, 0.45), xytext=(7.75, 3.7), arrowprops=dict(arrowstyle="-|>", color=C.ink, lw=1.2))
    label(ax, C, 7.55, 2.1, "authority", fs=8, rotation=90, color=C.ink)
    label(ax, C, 3.4, 0.12, "a hazard reaches the motors only if every layer fails", fs=8)
    return fig


@figure
def test_ladder(C):
    fig, ax = canvas(8, 3.2)
    steps = [("SITL", "real firmware", "blue"), ("HITL", "real FC timing", "aqua"), ("bench", "real hardware,\nprops off", "yellow"),
             ("tethered", "real dynamics", "orange"), ("field", "real environment", "red")]
    for k, (t, adds, hue) in enumerate(steps):
        x, y = 0.2 + k * 1.55, 0.35 + k * 0.48
        box(ax, C, x, y, 1.4, 0.55, t, hue, fs=9, weight="bold")
        label(ax, C, x + 0.7, y + 0.75, "+ " + adds, fs=7.5, color=C.ink)
    label(ax, C, 1.6, 2.6, "cost of a failure grows →\nreturn to SITL after every change", fs=8, color=C.muted)
    return fig


# ======================================================================= Chapter 14

def _robot(ax, C, x, y, t, hue="blue", r=0.22):
    ax.add_patch(Circle((x, y), r, fc=C.soft(hue), ec=C[hue], lw=1.2))
    ax.text(x, y, t, ha="center", va="center", fontsize=7.5, color=C.ink)


@figure
def team_architectures(C):
    fig, ax = canvas(8, 3.3)
    titles = ["centralised", "hierarchical / CTDE", "decentralised"]
    for p, t in enumerate(titles):
        label(ax, C, 1.3 + p * 2.7, 3.12, t, fs=9, color=C.ink, weight="bold")
    # centralised
    box(ax, C, 0.75, 2.05, 1.1, 0.55, "one policy\n(all obs)", "violet", fs=7.5)
    for k in range(3):
        x = 0.45 + k * 0.85
        _robot(ax, C, x, 0.75, f"R{k + 1}")
        arrow(ax, C, (x, 0.97), (1.3, 2.05), lw=0.8, color=C["aqua"])
        arrow(ax, C, (1.3, 2.05), (x + 0.05, 0.97), lw=0.8, color=C["violet"])
    label(ax, C, 1.3, 0.2, "obs up, actions down,\nevery step", fs=7)
    # hierarchical
    box(ax, C, 3.45, 2.05, 1.1, 0.55, "allocator\n(tasks)", "yellow", fs=7.5)
    for k in range(3):
        x = 3.15 + k * 0.85
        _robot(ax, C, x, 0.75, f"VLA{k + 1}", r=0.3)
        arrow(ax, C, (4.0, 2.05), (x, 1.05), lw=0.8, color=C["yellow"])
    label(ax, C, 4.0, 0.2, "tasks down per task;\nrobots act locally", fs=7)
    # decentralised
    pos = [(6.2, 1.9), (7.3, 1.9), (6.75, 0.85)]
    for k, (x, y) in enumerate(pos):
        _robot(ax, C, x, y, f"VLA{k + 1}", r=0.3)
    for a_, b_ in [(0, 1), (1, 2), (0, 2)]:
        (x0, y0), (x1, y1) = pos[a_], pos[b_]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="<|-|>", color=C["orange"], lw=0.9,
                                                                        ls="--", shrinkA=13, shrinkB=13))
    label(ax, C, 6.75, 0.2, "neighbours exchange\nmessages", fs=7)
    for x in (2.6, 5.35):
        ax.plot([x, x], [0.15, 3.0], color=C.faint, lw=1)
    return fig


@figure
def message_types(C):
    items = [("pose / state", 50, 1.0, "blue"), ("task bid / claim", 60, 1.6, "blue"), ("map / detection", 2_000, 2.6, "aqua"),
             ("learned vector", 500, 3.2, "violet"), ("natural language", 200, 4.2, "orange"),
             ("visual tokens (64 × 960, fp16)", 123_000, 4.3, "yellow"), ("image (JPEG)", 30_000, 4.9, "yellow")]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.axvspan(10, 2_500, color=C.soft("green", 0.18), lw=0)
    ax.text(14, 0.55, "fits a shared channel for 10 robots at 10 Hz (≈2 Mbit/s)", fontsize=7.5, color=C["green"])
    for t, b, e, hue in items:
        ax.plot(b, e, "o", ms=9, color=C[hue])
        ax.text(b * (1.25 if b < 20_000 else 0.8), e, t, fontsize=8, va="center", ha="left" if b < 20_000 else "right",
                color=C.ink)
    ax.set_xscale("log"); ax.set_xlim(10, 300_000); ax.set_ylim(0.3, 5.3)
    ax.set_xlabel("bytes per message"); ax.set_ylabel("expressiveness (qualitative)")
    ax.set_yticks([])
    return fig


@figure
def language_coordination(C):
    fig, ax = canvas(8, 3.5)
    cols = ["planner (SMART-LLM)", "negotiator (RoCo)", "communicator (CoELA)"]
    for p, t in enumerate(cols):
        label(ax, C, 1.3 + p * 2.7, 3.33, t, fs=9, color=C.ink, weight="bold")
    # planner
    box(ax, C, 0.45, 2.45, 1.7, 0.55, "LLM: decompose,\nform coalitions, allocate", "orange", fs=7)
    box(ax, C, 0.45, 1.6, 1.7, 0.45, "verifier", "red", fs=7.5)
    arrow(ax, C, (1.3, 2.45), (1.3, 2.05), lw=0.9)
    for k in range(3):
        x = 0.55 + k * 0.75
        _robot(ax, C, x, 0.85, "skill", "blue", r=0.25)
        arrow(ax, C, (1.3, 1.6), (x, 1.1), lw=0.8)
    # negotiator
    for k, (x, y) in enumerate([(3.4, 2.55), (4.6, 2.55)]):
        box(ax, C, x - 0.5, y - 0.25, 1.0, 0.5, f"LLM agent {k + 1}", "orange", fs=7)
    ax.annotate("", xy=(4.1, 2.55), xytext=(3.9, 2.55), arrowprops=dict(arrowstyle="<|-|>", color=C["orange"], lw=1))
    label(ax, C, 4.0, 2.95, "dialogue", fs=7, color=C["orange"])
    box(ax, C, 3.1, 1.6, 1.8, 0.45, "env. check + motion planner", "red", fs=7)
    arrow(ax, C, (4.0, 2.3), (4.0, 2.05), lw=0.9)
    for k in range(2):
        x = 3.5 + k * 1.0
        _robot(ax, C, x, 0.85, f"arm {k + 1}", "blue", r=0.27)
        arrow(ax, C, (4.0, 1.6), (x, 1.12), lw=0.8)
    # communicator
    for k, x in enumerate((6.1, 7.4)):
        box(ax, C, x - 0.55, 1.95, 1.1, 0.95, "perceive\nremember\nplan · talk", "orange", fs=7)
        _robot(ax, C, x, 0.85, f"robot {k + 1}", "blue", r=0.3)
        arrow(ax, C, (x, 1.95), (x, 1.15), lw=0.8)
    ax.annotate("", xy=(6.85, 2.45), xytext=(6.65, 2.45), arrowprops=dict(arrowstyle="<|-|>", color=C["orange"], lw=1))
    label(ax, C, 6.75, 3.02, "messages", fs=7, color=C["orange"])
    label(ax, C, 4.0, 0.2, "in every pattern, checks and per-robot policies stand between text and motors", fs=8)
    for x in (2.6, 5.35):
        ax.plot([x, x], [0.45, 3.2], color=C.faint, lw=1)
    return fig


@figure
def allocation(C):
    from vlacourse.teams import auction, brute_force, greedy, total
    robots = np.array([[0.12, 0.5], [0.5, 0.18], [0.88, 0.52]])
    tasks = np.array([[0.32, 0.5], [0.62, 0.55], [0.82, 0.86]])
    cost = np.linalg.norm(robots[:, None] - tasks[None], axis=-1)
    g, a = greedy(cost), auction(cost, eps=1e-6)[0]
    assert (a == brute_force(cost)).all()
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, asg, name in [(axs[0], g, "greedy"), (axs[1], a, "optimal (auction)")]:
        for i, j in enumerate(asg):
            ax.annotate("", xy=tasks[j], xytext=robots[i], arrowprops=dict(arrowstyle="-|>", color=C["violet"], lw=1.6))
        ax.plot(*robots.T, "o", ms=13, color=C["blue"])
        ax.plot(*tasks.T, "s", ms=11, color=C["orange"])
        for k, (x, y) in enumerate(robots):
            ax.text(x, y - 0.075, f"R{k + 1}", ha="center", fontsize=8, color=C.ink)
        for k, (x, y) in enumerate(tasks):
            ax.text(x, y + 0.06, f"T{k + 1}", ha="center", fontsize=8, color=C.ink)
        ax.set_title(f"{name}: total {total(cost, asg):.2f}, longest {total(cost, asg, makespan=True):.2f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0.05, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return fig


@figure
def coverage(C):
    G = 60
    xs = (np.arange(G) + 0.5) / G
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    phi = 0.15 + np.exp(-((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.02) + 0.8 * np.exp(-((X - 0.3) ** 2 + (Y - 0.75) ** 2) / 0.03)
    q = np.stack([X, Y], -1).reshape(-1, 2)
    w = phi.reshape(-1)
    p = np.array([[0.08, 0.08], [0.15, 0.1], [0.1, 0.18], [0.2, 0.15], [0.25, 0.08], [0.06, 0.28]])
    traj, costs = [p.copy()], []
    for _ in range(40):
        owner = np.argmin(((q[:, None] - p[None]) ** 2).sum(-1), 1)
        costs.append((((q - p[owner]) ** 2).sum(-1) * w).sum() / w.sum())
        new = p.copy()
        for i in range(len(p)):
            m = owner == i
            c = (q[m] * w[m, None]).sum(0) / w[m].sum()
            new[i] = p[i] + 0.5 * (c - p[i])
        p = new
        traj.append(p.copy())
    traj = np.array(traj)
    owner = np.argmin(((q[:, None] - p[None]) ** 2).sum(-1), 1).reshape(G, G)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.3))
    a.imshow(phi, origin="lower", extent=(0, 1, 0, 1), cmap=C.seq("blue"), alpha=0.9)
    edge = np.zeros_like(owner, bool)
    edge[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    edge[1:, :] |= owner[1:, :] != owner[:-1, :]
    a.imshow(np.ma.masked_where(~edge, edge), origin="lower", extent=(0, 1, 0, 1), cmap=plt.matplotlib.colors.ListedColormap([C.ink]),
             alpha=0.7, interpolation="nearest")
    for i in range(len(p)):
        a.plot(traj[:, i, 0], traj[:, i, 1], color=C["orange"], lw=1.2)
    a.plot(*traj[0].T, "o", ms=4, color=C.muted)
    a.plot(*p.T, "o", ms=8, color=C["orange"], mec=C.surface)
    a.set_xticks([]); a.set_yticks([]); a.set_title("paths to the weighted Voronoi centroids", fontsize=9)
    b.plot(costs, color=C["blue"])
    b.set_xlabel("iteration"); b.set_ylabel("coverage cost H"); b.set_title("cost falls monotonically", fontsize=9)
    return fig


@figure
def fleet_learning(C):
    fig, ax = canvas(8, 3.3)
    for k in range(4):
        _robot(ax, C, 0.55 + k * 0.75, 2.6, f"R{k + 1}", r=0.28)
    label(ax, C, 1.65, 3.15, "fleet running the shared policy", fs=8, color=C.ink)
    box(ax, C, 0.2, 1.2, 2.9, 0.55, "uncertainty / risk estimates", "violet", fs=8)
    for k in range(4):
        arrow(ax, C, (0.55 + k * 0.75, 2.32), (0.55 + k * 0.75, 1.75), lw=0.7)
    box(ax, C, 3.6, 1.2, 1.4, 0.55, "few human\nsupervisors", "orange", fs=8)
    arrow(ax, C, (3.1, 1.48), (3.6, 1.48))
    ax.annotate("", xy=(2.8, 2.45), xytext=(4.3, 1.75), arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1,
                                                                        connectionstyle="arc3,rad=0.3"))
    label(ax, C, 4.1, 2.45, "interventions", fs=7.5, color=C["orange"])
    box(ax, C, 5.5, 1.75, 2.3, 1.0, "shared dataset\nautonomous + corrected\nepisodes, outcomes,\nmessages, assignments", "yellow", fs=7)
    arrow(ax, C, (5.0, 1.48), (5.5, 1.9))
    box(ax, C, 5.5, 0.3, 2.3, 0.6, "retrain → redeploy", "aqua", fs=8)
    arrow(ax, C, (6.65, 1.75), (6.65, 0.9))
    ax.annotate("", xy=(1.65, 0.95), xytext=(5.5, 0.6), arrowprops=dict(arrowstyle="-|>", color=C["aqua"], lw=1.2,
                                                                         connectionstyle="arc3,rad=-0.15"))
    return fig


# ======================================================================= Appendix A

@figure
def mujoco_depot(C):
    import mujoco as _mj
    from vlacourse.mujoco_depot import MujocoDepotNav
    env = MujocoDepotNav(res=64)
    env.reset(seed=10)
    for _ in range(4):
        env.step(env.expert_action())
    big = _mj.Renderer(env.model, 256, 256)
    big.update_scene(env.data, camera="top")
    top = big.render().copy()
    big.update_scene(env.data, camera="onboard")
    onboard = big.render().copy()
    big.close()
    small = env.obs()["image"]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8), gridspec_kw=dict(wspace=0.08))
    for ax, im_, t in [(axs[0], top, "overhead camera, 256 px"), (axs[1], small, "what the toy VLA sees, 64 px"),
                       (axs[2], onboard, "onboard camera")]:
        ax.imshow(im_, interpolation="nearest" if im_.shape[0] == 64 else "antialiased")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.text(0.5, 0.0, f"instruction: '{env.instruction}'", ha="center", fontsize=9, color=C.muted)
    return fig


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    unknown = set(names) - FIGS.keys()
    if unknown:
        sys.exit(f"unknown figures: {sorted(unknown)}; known: {sorted(FIGS)}")
    for name in names or FIGS:
        for theme in THEMES:
            C = Theme(theme)
            style(C)
            fig = FIGS[name](C)
            suffix = "" if theme == "light" else "-dark"
            fig.savefig(OUT / f"{name}{suffix}.svg", bbox_inches="tight", pad_inches=0.06, metadata={"Date": None})
            plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
