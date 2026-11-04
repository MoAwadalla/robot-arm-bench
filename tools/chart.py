"""Draw media/chart.png from media/numbers.json (run tools/tally.py first).

    uv run --with matplotlib python tools/chart.py                      # media/numbers.json -> media/chart.png
    uv run --with matplotlib python tools/chart.py <folder>/numbers.json   # -> <folder>/chart.png

matplotlib is not a project dependency, hence --with.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6.1-sol": "GPT-6.1 Sol",
}
ORDER = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "gpt-6-luna", "gpt-6.1-sol"]                                  # same order as the README tables
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e8e7e3"
PICTURES, POSITIONS = "#2a78d6", "#eb6834"               # categorical slots 1 and 2


def bar(ax, x, w, h, color, r_px=4):
    """A bar with a rounded top and a square base."""
    if h <= 0:
        return
    # convert a 4 px radius into data units on each axis
    fig = ax.figure
    bbox = ax.get_window_extent()
    xr = (ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width * r_px * fig.dpi / 72 * 0.75
    yr = (ax.get_ylim()[1] - ax.get_ylim()[0]) / bbox.height * r_px * fig.dpi / 72 * 0.75
    xr, yr = min(xr, w / 2), min(yr, h)
    x0, x1, top = x - w / 2, x + w / 2, h
    verts = [(x0, 0), (x0, top - yr), (x0, top), (x0 + xr, top), (x1 - xr, top), (x1, top), (x1, top - yr),
             (x1, 0), (x0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))


def main() -> None:
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "media" / "numbers.json"
    data = json.loads(src.read_text())["levels_3_5"]
    cols = [c for c in ORDER if c in data and "average pictures" in data[c]]
    cols += sorted(c for c in data if c not in ORDER and "average pictures" in data[c] and "average positions" in data[c])
    pics = [100 * data[c]["average pictures"] for c in cols]
    pos = [100 * data[c].get("average positions", 0) for c in cols]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(0, 108)

    w, gap = 0.22, 0.025
    for i, (p, q) in enumerate(zip(pics, pos)):
        for x, h, color in ((i - w / 2 - gap / 2, p, PICTURES), (i + w / 2 + gap / 2, q, POSITIONS)):
            bar(ax, x, w, h, color)
            ax.text(x, h + 1.8, f"{round(h)}", ha="center", va="bottom", fontsize=9.5, color=INK2)

    ax.set_xticks(range(len(cols)), [LABELS.get(c, c) for c in cols], fontsize=10, color=INK)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Average score, levels 3–5 (%)", fontsize=10, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(axis="y", length=0, labelcolor=INK2, labelsize=9)
    ax.tick_params(axis="x", length=0, pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PICTURES), plt.Rectangle((0, 0), 1, 1, color=POSITIONS)]
    ax.legend(handles, ["From camera pictures only", "With the block positions given"], frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10, labelcolor=INK, handlelength=1.2,
              handleheight=1.2, borderaxespad=0.3)
    fig.text(0.012, 0.015, "3 layouts per level, one run per layout (Sonnet: two, one with each setting, averaged). "
             "Haiku and Opus: default settings. Luna and Sol: medium reasoning effort.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(src.parent / "chart.png", facecolor=SURFACE)
    print(f"wrote {src.parent / 'chart.png'}")


if __name__ == "__main__":
    main()
