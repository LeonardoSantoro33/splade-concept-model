from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# figures always go to plots/figures/, whatever folder the script is launched from
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = [256, 512, 1024, 2048, 4096, 8192]  # x axis

# --- Reference -------------------------------------------------------------------
SPLADE_DEV = 0.3828   # MS MARCO dev MRR@10
SPLADE_BEIR = 0.4662  # BEIR mean nDCG@10
SPLADE_GB = 8.5

# --- Concept Model alone ---------------------------------------------------------
DEV = {
    "Random W": [0.2544, 0.3129, 0.3454, 0.3634, 0.3691, 0.3761],
    "Frozen":   [0.2789, 0.3256, 0.3451, 0.3683, 0.3717, 0.3766],
    "e2e":      [0.3424, 0.3598, 0.3731, 0.3799, 0.3829, 0.3722],
}
BEIR = {
    "Random W": [0.3117, 0.3929, 0.4300, 0.4370, 0.4601, 0.4600],
    "Frozen":   [0.3356, 0.4016, 0.4294, 0.4431, 0.4598, 0.4605],
    "e2e":      [0.3372, 0.4027, 0.4291, 0.4347, 0.4580, 0.4541],
}

# --- Hybrids (lambda* chosen on MS MARCO validation) -----------------------------
HYB_DEV = {
    "Frozen hybrid": [0.3835, 0.3815, 0.3779, 0.3828, 0.3831, 0.3806],
    "e2e hybrid":    [0.3901, 0.3873, 0.3855, 0.3881, 0.3880, 0.3722],
    "hybnoW":        [0.3920, 0.3910, 0.3915, 0.3908, 0.3887, 0.3847],
}
HYB_BEIR = {
    "Frozen hybrid": [0.4659, 0.4642, 0.4604, 0.4662, 0.4660, 0.4617],
    "e2e hybrid":    [0.4646, 0.4608, 0.4510, 0.4630, 0.4661, 0.4541],
    "hybnoW":        [0.4676, 0.4646, 0.4654, 0.4555, 0.4617, 0.4646],
}

# --- Document storage (GB, raw sizes) ---------------------------------------------
E2E_GB = [4.5, 9.1, 18.1, 36.2, 72.4, 144.9]           # dense fp16 vectors
HYB_E2E_GB = [15.8, 16.7, 18.8, 20.3, 19.1, 16.0]      # SPLADE index + fine-tuned encoder index
HYB_FROZEN_GB = [8.53, 8.56, 8.63, 8.75, 9.00, 9.50]   # SPLADE index (8.5 GB) + W (fp32)

# --- Style -------------------------------------------------------------------------
# validated categorical palette (blue, orange, aqua): one fixed color per system
COLORS = {
    "Random W": "#2a78d6", "Frozen": "#eb6834", "e2e": "#1baf7a",
    "Frozen hybrid": "#eb6834", "e2e hybrid": "#1baf7a", "hybnoW": "#2a78d6",
    "e2e hybrid (space)": "#2a78d6",
}
MARKERS = {
    "Random W": "s", "Frozen": "o", "e2e": "^",
    "Frozen hybrid": "o", "e2e hybrid": "^", "hybnoW": "D",
    "e2e hybrid (space)": "D",
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": TEXT_2,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
})


def k_axis(ax):
    """Base-2 logarithmic x axis labelled with the k values."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("k (number of concepts)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def splade_line(ax, y, text="SPLADE"):
    ax.axhline(y, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=1)
    # label outside the data area, to the right of the line
    ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", color=TEXT_2, fontsize=9, annotation_clip=False)


def series(ax, x, y, name):
    ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], linewidth=2,
            markersize=6, markeredgecolor="white", markeredgewidth=1, label=name, zorder=3)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR / name}.png and .pdf")


# ===== Figure 1: Concept Model alone, % of SPLADE ===================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, data, ref, title in [
    (a1, DEV, SPLADE_DEV, "In-domain: MS MARCO dev (MRR@10)"),
    (a2, BEIR, SPLADE_BEIR, "Out-of-domain: BEIR (mean nDCG@10)"),
]:
    for name, values in data.items():
        series(ax, K, [100 * v / ref for v in values], name)
    splade_line(ax, 100, "SPLADE = 100%")
    k_axis(ax)
    ax.set_title(title, color=TEXT, fontsize=11)
a1.set_ylabel("% of SPLADE")
a1.legend(frameon=False, loc="lower right")
save(fig, "fig1_percent_of_splade")

# ===== Figure 2: hybrids, absolute values ===========================================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, data, ref, title, ylab in [
    (a1, HYB_DEV, SPLADE_DEV, "In-domain: MS MARCO dev", "MRR@10"),
    (a2, HYB_BEIR, SPLADE_BEIR, "Out-of-domain: BEIR", "mean nDCG@10"),
]:
    for name, values in data.items():
        series(ax, K, values, name)
    splade_line(ax, ref)
    k_axis(ax)
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.set_ylabel(ylab)
a1.legend(frameon=False, loc="lower left")
save(fig, "fig2_hybrids")

# ===== Figure 3: storage vs quality (in-domain) =====================================
fig, ax = plt.subplots(figsize=(7, 4.4))
series(ax, E2E_GB, DEV["e2e"], "e2e")
ax.lines[-1].set_label("e2e alone (dense fp16 vectors)")
ax.scatter(HYB_E2E_GB, HYB_DEV["e2e hybrid"], color=COLORS["e2e hybrid (space)"],
           marker="D", s=40, edgecolor="white", linewidth=1, zorder=4,
           label="e2e hybrid (SPLADE index + encoder index)")
for gb, v, k in zip(E2E_GB, DEV["e2e"], K):
    ax.annotate(f"k={k}", (gb, v), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color=TEXT_2)
# label positions for the hybrid points (they are very close to each other)
HYB_OFFSETS = {256: (-7, 3, "right"), 512: (-7, -9, "right"), 1024: (7, -9, "left"),
               2048: (7, 0, "left"), 4096: (0, 9, "center"), 8192: (-7, -9, "right")}
for gb, v, k in zip(HYB_E2E_GB, HYB_DEV["e2e hybrid"], K):
    dx, dy, ha = HYB_OFFSETS[k]
    ax.annotate(f"k={k}", (gb, v), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=8, color=TEXT_2)
ax.scatter([SPLADE_GB], [SPLADE_DEV], color=TEXT, marker="*", s=160, zorder=5,
           label="SPLADE")
ax.set_xscale("log")
ticks = [5, 10, 20, 50, 100, 150]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlim(3.5, 200)
ax.set_ylim(0.335, 0.395)
ax.set_xlabel("document storage (GB, log scale)")
ax.set_ylabel("MS MARCO dev MRR@10")
ax.set_title("Storage vs in-domain quality", color=TEXT, fontsize=11)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
save(fig, "fig3_space_vs_quality")

# ===== Figure 4: total document storage for each k ==================================
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(len(K))
width = 0.27
groups = [
    (E2E_GB, "#1baf7a", "Concept Model alone, frozen or e2e (dense fp16 vectors)"),
    (HYB_FROZEN_GB, "#eb6834", "Frozen hybrid (SPLADE index + W)"),
    (HYB_E2E_GB, "#2a78d6", "e2e hybrid (SPLADE index + encoder index)"),
]
for j, (values, color, name) in enumerate(groups):
    bars = ax.bar(x + (j - 1) * (width + 0.02), values, width, color=color, label=name, zorder=3)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7.5, color=TEXT_2)
ax.axhline(SPLADE_GB, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=2)
ax.annotate("SPLADE (8.5 GB)", xy=(1.0, SPLADE_GB),
            xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
            va="center", ha="left", color=TEXT_2, fontsize=9, annotation_clip=False)
ax.set_yscale("log")
ax.set_yticks([5, 10, 20, 50, 100, 150])
ax.set_yticklabels(["5", "10", "20", "50", "100", "150"])
ax.set_ylim(3.5, 260)
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in K])
ax.set_xlabel("k (number of concepts)")
ax.set_ylabel("document storage (GB, log scale)")
ax.set_title("Total document storage (8.84M passages)", color=TEXT, fontsize=11)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=9)
save(fig, "fig4_total_space")