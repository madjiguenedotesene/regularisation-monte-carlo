"""Partie 1 – Régularisation : Ridge, Lasso et Elastic Net face au surapprentissage.

Expérience du TP : 60 points d'une sinusoïde bruitée, ajustés par un polynôme de degré 15.
Ajout par rapport au TP : on mesure l'erreur sur des données NON vues (jeu de test simulé
et validation croisée), seule mesure qui révèle le surapprentissage.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from style import AQUA, BLEU, GRID, GRIS, INK, INK2, ORANGE, ROUGE, SURFACE, plt, sauver, titre

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

DEGRE = 15
BRUIT = 0.15
ALPHAS = np.logspace(-15, 1, 81)


def donnees_tp(graine=10):
    """Reproduit exactement les données du TP (60 points, graine 10)."""
    rng = np.random.RandomState(graine)
    x = np.array([i * np.pi / 180 for i in range(60, 300, 4)])
    y = np.sin(x) + rng.normal(0, BRUIT, len(x))
    return x, y


def donnees_test(n=2000, graine=123):
    """Nouvelles observations du même phénomène, jamais vues pendant l'apprentissage."""
    rng = np.random.RandomState(graine)
    x = rng.uniform(60 * np.pi / 180, 296 * np.pi / 180, n)
    return x, np.sin(x) + rng.normal(0, BRUIT, n)


def polynome(x, degre=DEGRE):
    return np.column_stack([x ** i for i in range(1, degre + 1)])


def modele(type_, alpha, l1_ratio=0.5):
    if type_ == "Ridge":
        reg = Ridge(alpha=alpha)
    elif type_ == "Lasso":
        reg = Lasso(alpha=alpha, max_iter=200_000, tol=1e-6)
    else:
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=200_000, tol=1e-6)
    return make_pipeline(StandardScaler(), reg)


def courbes_erreur(x, y, xt, yt):
    """Erreur d'apprentissage, de validation croisée et de test pour chaque alpha."""
    X, Xt = polynome(x), polynome(xt)
    cv = KFold(5, shuffle=True, random_state=0)
    lignes = []
    for type_ in ("Ridge", "Lasso"):
        for a in ALPHAS:
            m = modele(type_, a).fit(X, y)
            lignes.append({
                "modele": type_, "alpha": a,
                "mse_apprentissage": np.mean((m.predict(X) - y) ** 2),
                "mse_validation_croisee": -cross_val_score(modele(type_, a), X, y, cv=cv,
                                                           scoring="neg_mean_squared_error").mean(),
                "mse_test": np.mean((m.predict(Xt) - yt) ** 2),
                "coef_non_nuls": int(np.sum(np.abs(m[-1].coef_) > 1e-10)),
                "norme_coef": float(np.linalg.norm(m[-1].coef_)),
            })
    return pd.DataFrame(lignes)


def figure_tp(x, y, res):
    """Données du TP : trois ajustements + erreur d'apprentissage vs erreur de test."""
    r = res[res.modele == "Ridge"].reset_index(drop=True)
    a_opt = r.loc[r.mse_validation_croisee.idxmin(), "alpha"]
    cas = [(1e-15, "α = 10⁻¹⁵ : surapprentissage", ROUGE),
           (a_opt, f"α = {a_opt:.0e} : validation croisée".replace("e-0", "e-"), AQUA),
           (5.0, "α = 5 : sous-apprentissage", ORANGE)]

    fig = plt.figure(figsize=(14, 5.4))
    xs = np.linspace(x.min(), x.max(), 500)
    for k, (a, lib, c) in enumerate(cas):
        ax = fig.add_axes([0.04 + k * 0.165, 0.14, 0.145, 0.62])
        m = modele("Ridge", a).fit(polynome(x), y)
        ax.plot(xs, np.sin(xs), color=GRIS, lw=1.2, ls="--")
        ax.scatter(x, y, s=10, color=INK2, alpha=0.7)
        ax.plot(xs, m.predict(polynome(xs)), color=c, lw=2.2)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(lib, fontsize=9.5, color=INK)
        ax.set_xticks([])
        if k:
            ax.set_yticklabels([])
    fig.text(0.04, 0.07, "Points : données du TP · pointillés : vraie fonction sin(x) · trait : polynôme de degré 15 régularisé (Ridge)",
             fontsize=9, color=INK2)

    ax = fig.add_axes([0.58, 0.14, 0.40, 0.62])
    ax.plot(r.alpha, r.mse_apprentissage, color=GRIS, lw=2, label="Erreur sur les données d'apprentissage")
    ax.plot(r.alpha, r.mse_test, color=BLEU, lw=2.4, label="Erreur sur des données nouvelles (test)")
    ax.axhline(BRUIT ** 2, color=INK2, lw=0.8, ls=":")
    ax.text(1e-15, BRUIT ** 2 * 1.06, "bruit irréductible (σ² = 0,0225)", fontsize=8.5, color=INK2)
    for a, _, c in cas:
        v = r.loc[(r.alpha - a).abs().idxmin(), "mse_test"]
        ax.scatter([a], [v], s=70, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Force de la régularisation α (échelle log)")
    ax.set_ylabel("Erreur quadratique moyenne")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Erreur d'apprentissage contre erreur sur données nouvelles", fontsize=11)
    titre(fig, "Les 60 points du TP : des coefficients qui explosent, mais un surapprentissage modéré",
          "Polynôme de degré 15 · avec 60 points rapprochés, la courbe reste contrainte entre les points ; l'écart de test est faible")
    sauver(fig, "01_tp_60_points.png")
    return a_opt


def figure_chemins(x, y):
    """Chemins de régularisation : Ridge rétrécit, Lasso sélectionne."""
    X = polynome(x)
    alphas = np.logspace(-8, 1, 60)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for ax, type_, c in ((axes[0], "Ridge", BLEU), (axes[1], "Lasso", ORANGE)):
        coefs = np.array([modele(type_, a).fit(X, y)[-1].coef_ for a in alphas])
        for j in range(coefs.shape[1]):
            ax.plot(alphas, coefs[:, j], color=c, lw=1.1, alpha=0.35 + 0.65 * (j < 3))
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("α (échelle log)")
        ax.set_ylabel("Coefficients (échelle symlog)")
        if type_ == "Lasso":
            nz = [(np.abs(cf) > 1e-10).sum() for cf in coefs]
            ax2 = ax.twinx()
            ax2.step(alphas, nz, color=INK, lw=1.4, where="mid")
            ax2.set_ylabel("Variables conservées", color=INK)
            ax2.set_ylim(0, 16)
            ax2.grid(False)
            ax2.spines["right"].set_visible(True)
            ax.set_title("Lasso : les coefficients tombent à zéro, un par un", fontsize=11)
        else:
            ax.set_title("Ridge : tous les coefficients rétrécissent ensemble", fontsize=11)
    titre(fig, "Chemins de régularisation des 15 coefficients",
          "Ridge (L2) garde toutes les variables ; Lasso (L1) en élimine : c'est une sélection de variables automatique")
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight(), wspace=0.35)
    sauver(fig, "03_chemins_regularisation.png")


def elastic_net(x, y, xt, yt):
    """Grille alpha × l1_ratio de l'Elastic Net, évaluée sur le jeu de test."""
    X, Xt = polynome(x), polynome(xt)
    ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    alphas = np.logspace(-6, 0, 13)
    grille = np.zeros((len(ratios), len(alphas)))
    nz = np.zeros_like(grille, dtype=int)
    for i, r in enumerate(ratios):
        for j, a in enumerate(alphas):
            m = (modele("Ridge", a * len(y)) if r == 0 else modele("EN", a, r)).fit(X, y)
            grille[i, j] = np.mean((m.predict(Xt) - yt) ** 2)
            nz[i, j] = int(np.sum(np.abs(m[-1].coef_) > 1e-10))

    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(np.log10(grille), cmap="Blues_r", aspect="auto")
    for i in range(len(ratios)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{grille[i, j]:.3f}\n{nz[i, j]} var.".replace(".", ","), ha="center", va="center",
                    fontsize=7.2, color=SURFACE if np.log10(grille[i, j]) < np.log10(grille).mean() else INK)
    i_best, j_best = np.unravel_index(grille.argmin(), grille.shape)
    ax.add_patch(plt.Rectangle((j_best - 0.5, i_best - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))
    ax.set_xticks(range(len(alphas)), [f"{a:.0e}".replace("e-0", "e-").replace("e+00", "") for a in alphas], fontsize=8)
    ax.set_yticks(range(len(ratios)), ["0 (Ridge)", "0,2", "0,5", "0,8", "1 (Lasso)"])
    ax.set_xlabel("α")
    ax.set_ylabel("l1_ratio")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("log₁₀ de l'erreur de test")
    titre(fig, "Elastic Net : doser le mélange L1 / L2",
          f"Erreur de test et nombre de variables conservées ; meilleur réglage encadré "
          f"(α = {alphas[j_best]:.0e}, l1_ratio = {ratios[i_best]})".replace(".", ","))
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight())
    sauver(fig, "04_elastic_net.png")
    return pd.DataFrame(grille, index=ratios, columns=alphas), (alphas[j_best], ratios[i_best], grille.min())


def figure_erreurs(res):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, type_ in zip(axes, ("Ridge", "Lasso")):
        r = res[res.modele == type_]
        ax.plot(r.alpha, r.mse_apprentissage, color=GRIS, lw=2, label="Apprentissage")
        ax.plot(r.alpha, r.mse_validation_croisee, color=ORANGE, lw=2, label="Validation croisée (5 plis)")
        ax.plot(r.alpha, r.mse_test, color=BLEU, lw=2.2, label="Test (2 000 points nouveaux)")
        a_cv = r.loc[r.mse_validation_croisee.idxmin(), "alpha"]
        ax.axvline(a_cv, color=ORANGE, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("α (échelle log)")
        ax.set_title(f"{type_} : α retenu par validation croisée = {a_cv:.0e}".replace("e-0", "e-"), fontsize=11)
    axes[0].set_ylabel("Erreur quadratique moyenne")
    axes[0].legend(fontsize=9, loc="upper left")
    titre(fig, "La validation croisée retrouve l'α qui généralise le mieux",
          "Sans donnée nouvelle, la validation croisée estime l'erreur de test à partir des 60 points seulement")
    fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight(), wspace=0.08)
    sauver(fig, "02_erreurs_validation.png")


# --------------------------------------------------------------------------- Quand la régularisation compte-t-elle ?
GRILLE_CV = np.logspace(-12, 2, 43)


def ajuster(x, y, regularise):
    if regularise:
        return make_pipeline(StandardScaler(), RidgeCV(alphas=GRILLE_CV)).fit(polynome(x), y)
    return modele("Ridge", 1e-12).fit(polynome(x), y)


def experience_taille(xt, yt, tailles=(12, 15, 20, 30, 45, 60), tirages=200, graine=2026):
    """Erreur de test selon le nombre de points d'apprentissage, avec et sans régularisation."""
    rng = np.random.RandomState(graine)
    a, b = 60 * np.pi / 180, 296 * np.pi / 180
    Xt = polynome(xt)
    lignes = []
    for n in tailles:
        for t in range(tirages):
            x = np.sort(rng.uniform(a, b, n))
            y = np.sin(x) + rng.normal(0, BRUIT, n)
            for reg in (False, True):
                m = ajuster(x, y, reg)
                lignes.append({"n": n, "tirage": t, "regularise": reg,
                               "mse_test": float(np.mean((m.predict(Xt) - yt) ** 2))})
    return pd.DataFrame(lignes)


def exemple_representatif(exp, xt, yt, n_exemple=15, essais=300):
    """Choisit un échantillon dont les deux erreurs sont proches des médianes observées."""
    a, b = 60 * np.pi / 180, 296 * np.pi / 180
    cible = exp[exp.n == n_exemple].groupby("regularise")["mse_test"].median()
    Xt = polynome(xt)
    meilleur, score = None, np.inf
    for g in range(essais):
        rng = np.random.RandomState(g)
        x = np.sort(rng.uniform(a, b, n_exemple))
        y = np.sin(x) + rng.normal(0, BRUIT, n_exemple)
        e = {reg: np.mean((ajuster(x, y, reg).predict(Xt) - yt) ** 2) for reg in (False, True)}
        sc = abs(np.log(e[False] / cible[False])) + abs(np.log(e[True] / cible[True]))
        if sc < score:
            meilleur, score = (x, y), sc
    return meilleur


def figure_couverture(exp, xt, yt, n_exemple=15):
    """Image de couverture : la régularisation devient indispensable quand les données sont rares."""
    fig = plt.figure(figsize=(14, 5.6))
    a, b = 60 * np.pi / 180, 296 * np.pi / 180
    x, y = exemple_representatif(exp, xt, yt, n_exemple)
    xs = np.linspace(a, b, 600)

    for k, (reg, lib, c) in enumerate(((False, "Sans régularisation", ROUGE),
                                       (True, "Ridge, α choisi par validation croisée", AQUA))):
        ax = fig.add_axes([0.04 + k * 0.255, 0.13, 0.225, 0.62])
        m = ajuster(x, y, reg)
        mse = np.mean((m.predict(polynome(xt)) - yt) ** 2)
        ax.plot(xs, np.sin(xs), color=GRIS, lw=1.3, ls="--")
        ax.plot(xs, np.clip(m.predict(polynome(xs)), -3, 3), color=c, lw=2.3)
        ax.scatter(x, y, s=28, color=INK, zorder=3)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([])
        ax.set_title(f"{lib}\nerreur de test : {mse:.3f}".replace(".", ","), fontsize=10, color=INK)
        if k:
            ax.set_yticklabels([])
    fig.text(0.04, 0.06, f"Exemple typique avec {n_exemple} points · pointillés : vraie fonction sin(x) · trait : polynôme de degré 15",
             fontsize=9, color=INK2)

    ax = fig.add_axes([0.60, 0.13, 0.38, 0.62])
    g = exp.groupby(["n", "regularise"])["mse_test"]
    med, q1, q3 = g.median().unstack(), g.quantile(0.25).unstack(), g.quantile(0.75).unstack()
    for reg, c, lib in ((False, ROUGE, "Sans régularisation"), (True, AQUA, "Ridge + validation croisée")):
        ax.fill_between(med.index, q1[reg], q3[reg], color=c, alpha=0.15, lw=0)
        ax.plot(med.index, med[reg], color=c, lw=2.4, marker="o", ms=6, label=lib)
    ax.axhline(BRUIT ** 2, color=INK2, lw=0.8, ls=":")
    ax.text(med.index.min(), BRUIT ** 2 * 0.82, "bruit irréductible (σ² = 0,0225)", ha="left", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(list(med.index))
    ax.set_xlabel("Nombre de points d'apprentissage")
    ax.set_ylabel("Erreur de test (médiane, échelle log)")
    ax.legend(loc="upper right", fontsize=9)
    ratio = med.loc[min(med.index), False] / med.loc[min(med.index), True]
    ax.set_title(f"Avec {min(med.index)} points, l'erreur de test est divisée par " + f"{ratio:,.0f}".replace(",", " "),
                 fontsize=11)
    titre(fig, "La régularisation devient indispensable quand les données sont rares",
          f"Polynôme de degré 15 · {exp.tirage.nunique()} échantillons tirés par taille · bande : 50 % central des résultats")
    sauver(fig, "00_couverture.png")
    return med
