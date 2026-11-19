"""Charte graphique commune (palette catégorielle validée, fond clair)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLEU, ORANGE, AQUA, JAUNE, GRIS, ROUGE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#d03b3b"
BLEU_CLAIR = "#86b6ef"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200,
})


def titre(fig, texte, sous_titre=None):
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, texte, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    if sous_titre:
        fig.text(0.01, 1 - 0.47 / h, sous_titre, ha="left", va="top", fontsize=10.5, color=INK2)


def sauver(fig, nom):
    fig.savefig(f"docs/img/{nom}", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
