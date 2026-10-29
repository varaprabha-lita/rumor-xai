"""Paper figures (300-dpi PNG + vector PDF) into figures/.

  Figure 1  macro-F1 under random vs grouped split (leakage)
  Figure 2  pipeline diagram
  Figure 3  early detection curve
  Figure 4  confusion matrices of the LLM stage
Palette: categorical blue #2a78d6 / orange #eb6834 (validated light-mode pair);
grouped bars also carry hatching so the figure survives grayscale print.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R, F = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(F, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})
LABELS = ["non-rumor", "false", "true", "unverified"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load(ds, sp, key):
    return json.load(open(os.path.join(R, f"{ds}_{sp}_{key}.json")))["summary"]


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 120); ax.set_ylim(0, 60); ax.axis("off")

    def box(x, y, w, h, title, body, fill="#f3f2ee", edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=fill, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h - 2.5, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 7.5, body, ha="center", va="top", fontsize=6.5, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))

    box(1, 33, 20, 25, "Input event", "source post\n+ propagation tree\n+ reply text")
    box(28, 40, 22, 18, "1  BiGCN", "top-down and\nbottom-up graph\nconvolution\n→ verdict p_G")
    box(28, 12, 22, 22, "2  Node saliency", "gradient × node\nmask → top-5\nnodes, checked\nby deletion tests")
    box(57, 12, 26, 46, "3  Evidence prompt", "claim text\n+ cascade statistics\n+ GNN verdict p_G\n+ salient nodes\n(depth, arrival rank,\nreplies attracted)\n+ up to 10 replies\n(handles masked)")
    box(90, 31, 29, 27, "4  LLM", "Gemma 3 12B\ntemperature 0, JSON\n→ verdict p_L\n→ agrees with GNN?\n→ 3–5 sentence\nexplanation", fill="#e8f0fb", edge=BLUE)
    box(90, 1, 29, 24, "Evaluation", "random vs. grouped\n(leakage-free) splits\nmacro-F1, 95% CI,\nMcNemar test", fill="#fdf0ea", edge=ORANGE)
    arrow(21.6, 50, 27.4, 50); arrow(21.6, 40, 27.4, 26)
    arrow(39, 39.4, 39, 34.6)
    arrow(50.6, 49, 56.4, 45); arrow(50.6, 23, 56.4, 26)
    ax.plot([11, 11], [32.4, 5], color=INK2, lw=0.9); ax.plot([11, 70], [5, 5], color=INK2, lw=0.9); arrow(70, 5, 70, 11.4)
    ax.text(40, 6.2, "reply text", fontsize=6.5, color=INK2, ha="center", va="bottom")
    arrow(83.6, 44, 89.4, 44); arrow(104.5, 30.4, 104.5, 25.6)
    save(fig, "Figure2_pipeline")


# ------------------------------------------------------------------ Figure 2
def fig2():
    models = [("struct_rf", "Struct.\n+RF"), ("text_lr", "Text\n+LR"), ("text+struct_lr", "Text+\nstruct."),
              ("bigcn", "BiGCN"), ("fusion_tfidf", "BiGCN\n+text"), ("cascade_lr", "Cascade\nwords")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    x = np.arange(len(models)); w = 0.36
    for ax, ds in zip(axes, ("Twitter15", "Twitter16")):
        for j, (sp, col, hatch, lab) in enumerate((("random", BLUE, None, "Random split"),
                                                   ("group", ORANGE, "////", "Grouped (leakage-free) split"))):
            m = [load(ds, sp, k)["full"]["macro_f1"] for k, _ in models]
            v, e = np.array([a for a, _ in m]), np.array([b for _, b in m])
            bars = ax.bar(x + (j - 0.5) * (w + 0.03), v, w, yerr=e, color=col, edgecolor="white", linewidth=0.6,
                          hatch=hatch, error_kw=dict(lw=0.7, capsize=1.8, ecolor=INK2), label=lab, zorder=3)
            for b, val, err in zip(bars, v, e):
                ax.text(b.get_x() + b.get_width() / 2, val + err + 0.015, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=5.6, color=INK)
        ax.axhline(0.25, color=INK2, lw=0.8, ls="--", zorder=2, label="Chance (0.25)")
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in models], fontsize=7)
        ax.set_title(ds, fontsize=9, weight="bold", color=INK, loc="left")
        ax.set_ylim(0, 1.0); ax.set_xlim(-0.65, len(models) - 0.35); ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("Macro-F1 (5-fold mean ± s.d.)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Figure1_leakage")


# ------------------------------------------------------------------ Figure 3
def fig3():
    ks = ["k0", "k5", "k10", "k20", "full"]; xt = ["0", "5", "10", "20", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, ds in zip(axes, ("Twitter15", "Twitter16")):
        for sp, col, mk, lab in (("random", BLUE, "o", "Random split"), ("group", ORANGE, "s", "Grouped split")):
            s = load(ds, sp, "bigcn")
            v = [s[k]["macro_f1"][0] for k in ks]
            ax.plot(range(5), v, color=col, lw=2, marker=mk, ms=5, mec="white", mew=1, label=lab, zorder=3)
            ax.text(4.12, v[-1], f"{v[-1]:.2f}", va="center", fontsize=7, color=INK)
            ax.text(-0.12, v[0], f"{v[0]:.2f}", va="center", ha="right", fontsize=7, color=INK)
        ax.axhline(0.25, color=INK2, lw=0.8, ls="--"); ax.text(0.05, 0.265, "chance", fontsize=6.8, color=INK2)
        ax.set_xticks(range(5)); ax.set_xticklabels(xt); ax.set_xlim(-0.6, 4.6)
        ax.set_xlabel("Responses available at test time (k)")
        ax.set_title(ds, fontsize=9, weight="bold", color=INK, loc="left")
        ax.set_ylim(0.2, 0.9); ax.yaxis.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel("BiGCN macro-F1"); axes[1].legend(loc="center right", fontsize=7.2)
    fig.tight_layout()
    save(fig, "Figure3_early_detection")


# ------------------------------------------------------------------ Figure 4
def fig4():
    st = json.load(open(os.path.join(R, "llm_stats_gemma3_12b.json")))
    methods = [("BiGCN (grouped, out-of-fold)", "BiGCN"), ("LLM zero-shot, source", "LLM, source only"),
               ("LLM + propagation evidence", "LLM + propagation evidence")]
    short = ["NR", "F", "T", "U"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f7f9fd", "#9cc0ec", BLUE, "#0d3a73"])
    for r, ds in enumerate(("Twitter15", "Twitter16")):
        for c, (key, name) in enumerate(methods):
            ax = axes[r, c]; m = np.array(st[ds]["methods"][key]["confusion"])
            ax.imshow(m, cmap=cmap, vmin=0, vmax=50)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8,
                            color="white" if m[i, j] > 28 else INK, weight="bold" if i == j else "normal")
            ax.add_patch(plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, ec=ORANGE, lw=1.8))  # true -> non-rumour
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_xticklabels(short); ax.set_yticklabels(short)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            f1 = st[ds]["methods"][key]["macro_f1"]
            ax.set_title(f"{ds} · {name}\nmacro-F1 {f1:.3f}", fontsize=7.6, color=INK)
            if c == 0:
                ax.set_ylabel("Gold label")
            if r == 1:
                ax.set_xlabel("Predicted label")
    fig.text(0.5, -0.01, "NR = non-rumour, F = false, T = true, U = unverified (50 events per gold class). "
             "Orange box: true rumours predicted as non-rumours.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "Figure4_confusion")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
