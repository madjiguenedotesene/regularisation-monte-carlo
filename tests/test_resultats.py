"""Tests rapides des propriétés démontrées par le projet (pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import monte_carlo as mc  # noqa: E402
import regularisation as rg  # noqa: E402


def test_donnees_identiques_au_tp():
    x, y = rg.donnees_tp()
    assert len(x) == 60
    m = rg.modele("Ridge", 1e-4).fit(rg.polynome(x), y)
    rss = np.sum((m.predict(rg.polynome(x)) - y) ** 2)
    assert abs(rss - 0.9537) < 1e-3          # valeur du rapport du TP


def test_lasso_selectionne_des_variables():
    x, y = rg.donnees_tp()
    nz = np.count_nonzero(rg.modele("Lasso", 1e-3).fit(rg.polynome(x), y)[-1].coef_)
    assert 0 < nz < 15
    assert np.count_nonzero(rg.modele("Lasso", 1.0).fit(rg.polynome(x), y)[-1].coef_) == 0


def test_regularisation_utile_avec_peu_de_points():
    xt, yt = rg.donnees_test(n=500)
    exp = rg.experience_taille(xt, yt, tailles=(12,), tirages=30)
    med = exp.groupby("regularise")["mse_test"].median()
    assert med[True] < med[False] / 10


def test_ic_monte_carlo_contient_mu():
    r = mc.experience_tp()
    assert r["ic_bas"] <= mc.MU <= r["ic_haut"]


def test_couverture_proche_de_95_pourcent():
    cov = mc.couverture(tailles=(1000,), repetitions=2000)
    assert 0.94 <= cov.couverture.iloc[0] <= 0.96


def test_student_corrige_les_petits_echantillons():
    cov = mc.couverture(tailles=(10,), repetitions=4000)
    assert cov.couverture.iloc[0] < 0.935 < cov.couverture_student.iloc[0]
