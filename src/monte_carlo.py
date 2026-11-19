"""Partie 2 – Monte-Carlo et intervalles de confiance.

Le TP estimait la moyenne d'une loi N(10, 2²) avec 10 000 tirages. On vérifie ici
expérimentalement les trois propriétés annoncées :
  1. convergence de la moyenne empirique (loi des grands nombres) ;
  2. largeur de l'IC proportionnelle à 1/√N ;
  3. couverture : l'IC à 95 % contient la vraie moyenne dans 95 % des répétitions.
"""

import numpy as np
import pandas as pd
from scipy import stats

from style import AQUA, BLEU, BLEU_CLAIR, GRIS, INK, INK2, ORANGE, ROUGE, SURFACE, plt, sauver, titre

MU, SIGMA = 10.0, 2.0
Z = stats.norm.ppf(0.975)  # 1,96


def ic_moyenne(tirages, niveau=0.95):
    """Intervalle de confiance de la moyenne (approximation normale, σ estimé)."""
    n = len(tirages)
    m, s = tirages.mean(), tirages.std(ddof=1)
    z = stats.norm.ppf(0.5 + niveau / 2)
    return m, m - z * s / np.sqrt(n), m + z * s / np.sqrt(n)


def experience_tp(n=10_000, graine=42):
    rng = np.random.default_rng(graine)
    t = rng.normal(MU, SIGMA, n)
    m, bas, haut = ic_moyenne(t)
    return {"N": n, "moyenne": m, "variance": t.var(ddof=1), "ic_bas": bas, "ic_haut": haut,
            "contient_mu": bas <= MU <= haut}


def convergence(n_max=10_000, graine=42):
    rng = np.random.default_rng(graine)
    t = rng.normal(MU, SIGMA, n_max)
    n = np.arange(1, n_max + 1)
    moy = np.cumsum(t) / n
    demi = Z * SIGMA / np.sqrt(n)
    return n, moy, demi


def couverture(tailles=(10, 30, 100, 1000, 10_000), repetitions=2000, graine=7):
    """Part des IC qui contiennent la vraie moyenne, pour plusieurs tailles d'échantillon."""
    rng = np.random.default_rng(graine)
    lignes = []
    for n in tailles:
        t = rng.normal(MU, SIGMA, (repetitions, n))
        m, s = t.mean(axis=1), t.std(axis=1, ddof=1)
        demi = Z * s / np.sqrt(n)
        dedans = (m - demi <= MU) & (MU <= m + demi)
        p = dedans.mean()
        e = np.sqrt(p * (1 - p) / repetitions)
        demi_t = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
        p_t = ((m - demi_t <= MU) & (MU <= m + demi_t)).mean()
        e_t = np.sqrt(p_t * (1 - p_t) / repetitions)
        lignes.append({"N": n, "couverture": p, "ic_couverture_bas": p - Z * e, "ic_couverture_haut": p + Z * e,
                       "couverture_student": p_t, "ic_student_bas": p_t - Z * e_t, "ic_student_haut": p_t + Z * e_t,
                       "largeur_moyenne_ic": (2 * demi).mean()})
    return pd.DataFrame(lignes)


def figure_convergence(n, moy, demi):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(n, MU - demi, MU + demi, color=BLEU_CLAIR, alpha=0.45, lw=0,
                    label="Zone de fluctuation à 95 % (±1,96 σ/√N)")
    ax.plot(n, moy, color=BLEU, lw=1.6, label="Moyenne empirique après N tirages")
    ax.axhline(MU, color=ORANGE, lw=1.4, ls="--", label="Vraie moyenne μ = 10")
    ax.set_xscale("log")
    ax.set_ylim(MU - 3, MU + 3)
    ax.set_xlabel("Nombre de tirages N (échelle log)")
    ax.set_ylabel("Moyenne estimée")
    ax.legend(loc="upper right", fontsize=9)
    titre(fig, "Loi des grands nombres : la moyenne converge vers μ",
          "Une trajectoire de 10 000 tirages de N(10, 2²) ; la zone bleue se resserre en 1/√N")
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight())
    sauver(fig, "05_convergence.png")


def figure_couverture_ic(cov, graine=3):
    rng = np.random.default_rng(graine)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    # 100 intervalles de confiance (N = 30)
    n, k = 30, 100
    t = rng.normal(MU, SIGMA, (k, n))
    m, s = t.mean(axis=1), t.std(axis=1, ddof=1)
    demi = Z * s / np.sqrt(n)
    rate = ~((m - demi <= MU) & (MU <= m + demi))
    for i in range(k):
        c = ROUGE if rate[i] else BLEU
        a1.plot([i, i], [m[i] - demi[i], m[i] + demi[i]], color=c, lw=1.6 if rate[i] else 1.1)
        a1.scatter([i], [m[i]], s=7, color=c)
    a1.axhline(MU, color=ORANGE, lw=1.4, ls="--")
    a1.set_xlabel("Répétition de l'expérience (N = 30 tirages chacune)")
    a1.set_ylabel("IC à 95 % de la moyenne")
    a1.set_title(f"{k - rate.sum()} intervalles sur {k} contiennent μ = 10 ({rate.sum()} en rouge le manquent)", fontsize=11)
    a1.grid(axis="x", visible=False)

    # Couverture mesurée sur 2 000 répétitions
    x = np.arange(len(cov))
    a2.axhline(0.95, color=ORANGE, lw=1.4, ls="--")
    a2.text(-0.4, 0.9505, "95 % attendus", ha="left", va="bottom", fontsize=9, color=ORANGE)
    d = 0.12
    a2.errorbar(x - d, cov.couverture, yerr=[cov.couverture - cov.ic_couverture_bas, cov.ic_couverture_haut - cov.couverture],
                fmt="o", color=BLEU, ms=6, capsize=3, lw=1.3, label="IC normal (z = 1,96)")
    a2.errorbar(x + d, cov.couverture_student, yerr=[cov.couverture_student - cov.ic_student_bas,
                                                     cov.ic_student_haut - cov.couverture_student],
                fmt="s", color=AQUA, ms=6, capsize=3, lw=1.3, label="IC de Student (t)")
    a2.text(x[0] - d - 0.1, cov.couverture[0], f"{cov.couverture[0]:.1%}".replace(".", ",").replace("%", " %"),
            ha="right", va="center", fontsize=9, color=INK)
    a2.legend(loc="lower right", fontsize=9)
    a2.set_xticks(x, [f"{n:,}".replace(",", " ") for n in cov.N])
    a2.set_xlabel("Taille de l'échantillon N")
    a2.set_ylabel("Part des IC contenant μ")
    a2.set_ylim(0.895, 0.97)
    a2.set_xlim(-0.6, len(cov) - 0.4)
    a2.set_title("Couverture mesurée sur 2 000 répétitions", fontsize=11)
    titre(fig, "« 95 % de confiance » vérifié par simulation",
          "Avec peu de données, l'IC « 1,96 » du TP est trop étroit (91,5 % à N = 10) ; la loi de Student corrige ce biais")
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight(), wspace=0.25)
    sauver(fig, "06_couverture_ic.png")


def figure_largeur(cov):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.loglog(cov.N, cov.largeur_moyenne_ic, "o-", color=BLEU, lw=2, ms=7, label="Largeur moyenne mesurée")
    nn = np.logspace(1, 4, 50)
    ax.loglog(nn, 2 * Z * SIGMA / np.sqrt(nn), color=ORANGE, lw=1.4, ls="--", label="Théorie : 2 × 1,96 σ / √N")
    pente = np.polyfit(np.log(cov.N), np.log(cov.largeur_moyenne_ic), 1)[0]
    ax.set_xlabel("Taille de l'échantillon N")
    ax.set_ylabel("Largeur de l'IC à 95 %")
    ax.legend(fontsize=9)
    titre(fig, "×100 tirages → précision ×10",
          f"Pente mesurée en log-log : {pente:.3f} (théorie : −0,5)".replace(".", ","))
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight())
    sauver(fig, "07_largeur_ic.png")
    return pente
