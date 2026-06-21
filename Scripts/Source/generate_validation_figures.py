"""Generate sliding-window and expanding-window validation figures.

Designed to closely match the style of the original sliding-window-validation.pdf:
- Schematic (not to-scale) proportions
- Simple square-corner rectangles
- Chevron "move forward" arrows
- Curly-brace step-size annotations between rows
- (a)/(b) labels above top bar, (e)/(f) labels below bottom bar
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from pathlib import Path as PPath

OUT_DIR = PPath(__file__).resolve().parents[2] / "Scripts" / "Results"

# ── Colours (matching original PDF) ──────────────────────────────────────────
C_BLUE   = "#b8cce4"   # train blocks
C_ORANGE = "#f4a538"   # val / test blocks
C_EDGE   = "#222222"
C_TEXT   = "#222222"

# ── Schematic layout constants ────────────────────────────────────────────────
# X: 0..10   Y: varies per figure
# Full bar spans [0, 10]; split at x=8 (80/20).
# Windows are schematic: each window is 4 units wide (labeled "20%").

FULL_X0  = 0.0
FULL_X1  = 10.0
SPLIT    = 8.0          # 80/20 split
BAR_H    = 0.55

WIN_W    = 4.0           # schematic window width
TRAIN_W  = WIN_W * 0.80  # 3.2  (labeled "80%")
VAL_W    = WIN_W * 0.20  # 0.8  (labeled "20%")
STEP     = VAL_W          # 0.8
WIN_H    = 0.55
ROW_GAP  = 1.15           # vertical distance between row tops
N_FOLDS  = 4
WIN_X0   = 0.30           # left margin for the first window row


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _rect(ax, x, y, w, h, fc, label="", fs=11, bold=False, lw=1.5):
    p = mpatches.Rectangle((x, y), w, h,
                            facecolor=fc, edgecolor=C_EDGE, linewidth=lw, zorder=3)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=C_TEXT, zorder=4)


def _chevron(ax, x, y, w, h, gray, label="move forward", fs=14):
    """Fat right-pointing chevron arrow."""
    tip_indent = h * 0.55   # how deep the arrowhead indents the body
    notch      = h * 0.20   # back-notch depth
    my = y + h / 2

    verts = [
        (x + notch, y),
        (x + w - tip_indent, y),
        (x + w, my),
        (x + w - tip_indent, y + h),
        (x + notch, y + h),
        (x, my),
        (x + notch, y),
    ]
    fc = str(gray)   # matplotlib accepts float string as gray level
    poly = mpatches.Polygon(verts, closed=True,
                            facecolor=fc, edgecolor=C_EDGE, linewidth=1.2, zorder=3)
    ax.add_patch(poly)
    ax.text(x + (w - tip_indent) / 2, my, label,
            ha="center", va="center", fontsize=fs, fontweight="bold",
            color=C_TEXT, zorder=4)


def _top_brace(ax, x0, x1, y_base, label="", label_x=None, label_y=None,
               label_ha="left", fs=17, bold=True):
    """Downward-pointing bracket ⌐___¬ with optional label above."""
    mid = (x0 + x1) / 2
    th  = 0.18   # tick height

    verts = [
        (x0, y_base + th), (x0, y_base),
        (mid, y_base), (mid, y_base - th * 0.6),
        (mid, y_base), (x1, y_base),
        (x1, y_base + th),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * 6
    ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                 facecolor="none", edgecolor=C_EDGE, lw=1.3, zorder=5))
    if label:
        lx = label_x if label_x is not None else x0
        ly = label_y if label_y is not None else y_base + th + 0.08
        ax.text(lx, ly, label, ha=label_ha, va="bottom", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=C_TEXT, zorder=5)


def _step_brace(ax, x, y_lo, y_hi):
    """Vertical } brace pointing right, between two row y-values."""
    mid = (y_lo + y_hi) / 2
    bw  = 0.18

    verts = [
        (x, y_hi),
        (x + bw, y_hi), (x + bw, mid + 0.07),
        (x + bw * 1.9, mid),
        (x + bw, mid - 0.07), (x + bw, y_lo),
        (x, y_lo),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * 6
    ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                 facecolor="none", edgecolor=C_EDGE, lw=1.2, zorder=5))


def _vert_label_brace(ax, x, y_lo, y_hi, label, fs=9):
    """Vertical { brace on the left + rotated italic bold label."""
    mid = (y_lo + y_hi) / 2
    bw  = 0.22

    verts = [
        (x, y_hi),
        (x - bw, y_hi), (x - bw, mid + 0.07),
        (x - bw * 1.9, mid),
        (x - bw, mid - 0.07), (x - bw, y_lo),
        (x, y_lo),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * 6
    ax.add_patch(mpatches.PathPatch(Path(verts, codes),
                 facecolor="none", edgecolor=C_EDGE, lw=1.3, zorder=5))

    ax.text(x - bw * 2.2, mid, label,
            ha="right", va="center", fontsize=fs, color=C_TEXT,
            fontstyle="normal", fontweight="bold",
            rotation=90, rotation_mode="anchor", zorder=5)


def _save(fig, name):
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def make_sliding_window_figure():
    Y_TOP = 8.2
    Y_R0  = Y_TOP - BAR_H - 1.05    # bottom-y of first window row
    Y_BOT = Y_R0 - (N_FOLDS - 1) * ROW_GAP - WIN_H - 0.7

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(Y_BOT - 0.85, Y_TOP + BAR_H + 0.85)
    ax.axis("off")

    # ── top full bar ──────────────────────────────────────────────────────────
    _rect(ax, FULL_X0, Y_TOP, SPLIT,           BAR_H, C_BLUE,   "80% train set", fs=18, bold=True)
    _rect(ax, SPLIT,   Y_TOP, FULL_X1 - SPLIT, BAR_H, C_ORANGE, "20% test set",  fs=18, bold=True)

    # ── top braces ────────────────────────────────────────────────────────────
    brace_y = Y_R0 + WIN_H + 0.30

    # window-size brace: spans first window
    _top_brace(ax, WIN_X0, WIN_X0 + WIN_W, brace_y,
               label="window size",
               label_x=WIN_X0, label_ha="left", fs=17)

    # sliding window brace: spans the slide region, ending at the train-set edge
    slide_x0 = WIN_X0 + WIN_W + 0.15
    _top_brace(ax, slide_x0, SPLIT, brace_y,
               label="sliding window",
               label_x=(slide_x0 + SPLIT) / 2,
               label_ha="center", fs=17)

    # ── window rows ───────────────────────────────────────────────────────────
    offsets = [i * STEP for i in range(N_FOLDS)]

    for i, off in enumerate(offsets):
        y  = Y_R0 - i * ROW_GAP
        x  = WIN_X0 + off

        _rect(ax, x,          y, TRAIN_W, WIN_H, C_BLUE,   "80%", fs=18, bold=True)
        _rect(ax, x + TRAIN_W, y, VAL_W,  WIN_H, C_ORANGE, "20%", fs=18, bold=True)

    # ── single step-size indicator ──────────────────────────────────────────────
    # The window shifts right by one validation block (step size == val window).
    gap_y  = Y_R0 - (ROW_GAP - WIN_H) / 2          # mid-gap between rows 0 and 1
    x_lo   = WIN_X0                                  # left edge of row 0
    x_hi   = WIN_X0 + STEP                           # left edge of row 1
    # guide ticks tying the measure to the two windows' left edges
    ax.plot([x_lo, x_lo], [Y_R0, gap_y], color=C_EDGE, lw=1.0, zorder=4)
    ax.plot([x_hi, x_hi], [Y_R0 - ROW_GAP + WIN_H, gap_y], color=C_EDGE, lw=1.0, zorder=4)
    ax.annotate("", xy=(x_hi, gap_y), xytext=(x_lo, gap_y),
                arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=1.6), zorder=5)
    ax.text(x_hi + 0.20, gap_y, "step size", ha="left", va="center",
            fontsize=16, fontweight="bold", color=C_TEXT, zorder=5)

    # ── bottom full bar ───────────────────────────────────────────────────────
    _rect(ax, FULL_X0, Y_BOT, SPLIT,           BAR_H, C_BLUE,   "80% train set", fs=18, bold=True)
    _rect(ax, SPLIT,   Y_BOT, FULL_X1 - SPLIT, BAR_H, C_ORANGE, "20% test set",  fs=18, bold=True)
    ax.text(SPLIT / 2,             Y_BOT - 0.12, "Retrain with new hyperparameters",
            ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text((SPLIT + FULL_X1) / 2, Y_BOT - 0.12, "Final test",
            ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout(pad=0.2)
    _save(fig, "fig_sliding_window")


# ─────────────────────────────────────────────────────────────────────────────
# EXPANDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def make_expanding_window_figure():
    # EW: train always starts at WIN_X0 and grows each fold. The first fold is the
    # initial window (width WIN_W); the last fold validates up to the train-set
    # boundary (SPLIT), so the windows expand across the whole 80% train set.
    first_end = WIN_X0 + WIN_W      # initial window right edge (train + val)
    last_end  = SPLIT               # last fold validates up to the train-set edge
    val_ends  = [first_end + i * (last_end - first_end) / (N_FOLDS - 1)
                 for i in range(N_FOLDS)]
    train_ws  = [ve - VAL_W - WIN_X0 for ve in val_ends]
    val_x     = [WIN_X0 + tw for tw in train_ws]                # where val starts

    Y_TOP = 8.2
    Y_R0  = Y_TOP - BAR_H - 1.05
    Y_BOT = Y_R0 - (N_FOLDS - 1) * ROW_GAP - WIN_H - 0.7

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(Y_BOT - 0.85, Y_TOP + BAR_H + 0.85)
    ax.axis("off")

    # ── top full bar ──────────────────────────────────────────────────────────
    _rect(ax, FULL_X0, Y_TOP, SPLIT,           BAR_H, C_BLUE,   "80% train set", fs=18, bold=True)
    _rect(ax, SPLIT,   Y_TOP, FULL_X1 - SPLIT, BAR_H, C_ORANGE, "20% test set",  fs=18, bold=True)

    # ── top braces ────────────────────────────────────────────────────────────
    brace_y = Y_R0 + WIN_H + 0.30
    # Initial window brace: spans first window
    _top_brace(ax, WIN_X0, WIN_X0 + WIN_W, brace_y,
               label="initial window size",
               label_x=WIN_X0, label_ha="left", fs=17)

    # "expanding window" brace: spans the growth region up to the train-set edge
    max_val_end = SPLIT                            # last fold ends at train-set edge
    _top_brace(ax, WIN_X0 + WIN_W, max_val_end, brace_y,
               label="expanding window",
               label_x=(WIN_X0 + WIN_W + max_val_end) / 2,
               label_ha="center", fs=17)

    # ── window rows ───────────────────────────────────────────────────────────
    for i, (tw, vx) in enumerate(zip(train_ws, val_x)):
        y = Y_R0 - i * ROW_GAP
        _rect(ax, WIN_X0, y, tw,    WIN_H, C_BLUE,   "80%", fs=18, bold=True)
        _rect(ax, vx,     y, VAL_W, WIN_H, C_ORANGE, "20%", fs=18, bold=True)

    # ── bottom full bar ───────────────────────────────────────────────────────
    _rect(ax, FULL_X0, Y_BOT, SPLIT,           BAR_H, C_BLUE,   "80% train set", fs=18, bold=True)
    _rect(ax, SPLIT,   Y_BOT, FULL_X1 - SPLIT, BAR_H, C_ORANGE, "20% test set",  fs=18, bold=True)
    ax.text(SPLIT / 2,             Y_BOT - 0.12, "Retrain with new hyperparameters",
            ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text((SPLIT + FULL_X1) / 2, Y_BOT - 0.12, "Final test",
            ha="center", va="top", fontsize=15, fontweight="bold")

    fig.tight_layout(pad=0.2)
    _save(fig, "fig_expanding_window")


if __name__ == "__main__":
    make_sliding_window_figure()
    make_expanding_window_figure()
    print("Done.")
