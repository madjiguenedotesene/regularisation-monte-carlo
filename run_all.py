"""Pipeline complet : python run_all.py (depuis la racine du dépôt)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import monte_carlo as mc  # noqa: E402
import regularisation as rg  # noqa: E402

Path("docs/img").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("Partie 1 – Régularisation")
x, y = rg.donnees_tp()
xt, yt = rg.donnees_test()
res = rg.courbes_erreur(x, y, xt, yt)
res.to_csv("results/regularisation_erreurs.csv", index=False)
a_opt = rg.figure_tp(x, y, res)
exp = rg.experience_taille(xt, yt)
med = rg.figure_couverture(exp, xt, yt)
med.columns = ["sans_regularisation", "ridge_validation_croisee"]
med.to_csv("results/erreur_test_selon_taille.csv")
print(med.round(4))
rg.figure_erreurs(res)
rg.figure_chemins(x, y)
grille, meilleur_en = rg.elastic_net(x, y, xt, yt)
grille.to_csv("results/elastic_net_mse_test.csv")

synthese = {}
for type_ in ("Ridge", "Lasso"):
    r = res[res.modele == type_]
    cv_best = r.loc[r.mse_validation_croisee.idxmin()]
    faible = r.loc[(r.alpha - 1e-15).abs().idxmin()]
    synthese[type_] = {
        "alpha_validation_croisee": float(cv_best.alpha),
        "mse_test_alpha_retenu": round(float(cv_best.mse_test), 4),
        "mse_test_sans_regularisation": round(float(faible.mse_test), 4),
        "mse_apprentissage_sans_regularisation": round(float(faible.mse_apprentissage), 4),
        "variables_conservees": int(cv_best.coef_non_nuls),
    }
synthese["taille_echantillon"] = {int(k): {"sans": round(float(v.sans_regularisation), 4), "ridge_cv": round(float(v.ridge_validation_croisee), 4)} for k, v in med.iterrows()}
synthese["ElasticNet"] = {"alpha": float(meilleur_en[0]), "l1_ratio": meilleur_en[1], "mse_test": round(float(meilleur_en[2]), 4)}
for k, v in synthese.items():
    print(f"  {k}: {v}")

print("Partie 2 – Monte-Carlo")
tp = mc.experience_tp()
print(f"  N = {tp['N']} : moyenne {tp['moyenne']:.4f}, IC 95 % [{tp['ic_bas']:.4f} ; {tp['ic_haut']:.4f}]")
n, moy, demi = mc.convergence()
mc.figure_convergence(n, moy, demi)
cov = mc.couverture()
print(cov.round(4).to_string(index=False))
mc.figure_couverture_ic(cov)
pente = mc.figure_largeur(cov)
cov.to_csv("results/monte_carlo_couverture.csv", index=False)

synthese["MonteCarlo"] = {"experience_tp": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tp.items()},
                          "pente_largeur_ic": round(float(pente), 4)}
Path("results/synthese.json").write_text(json.dumps(synthese, indent=2, ensure_ascii=False, default=bool), encoding="utf-8")
print("Terminé : figures dans docs/img, tableaux dans results/")
