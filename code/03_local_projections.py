"""
03_local_projections.py
------------------------
Local Projections (Jorda 2005) de la reponse de l'inflation a un choc de
temperature, avec erreurs-types HAC (Newey-West). Deux apports :

(1) Estimation sur echantillon RECENT etendu (1987-2026), incluant l'episode
    inflationniste 2021-2023 - impossible avec le SVAR (activite reelle limitee a 2015).
(2) Test d'INTENSIFICATION : interaction du choc avec une indicatrice "ere recente"
    (annee >= 2007) pour voir si la transmission climat -> inflation s'est renforcee.

Specification, pour chaque horizon h :
    infl_{t+h} = a_h + b_h * temp_shock_t
                 + g_h * (temp_shock_t * late_t)
                 + controles (retards de infl, temp_shock, dloil, dy10) + e
b_h        : reponse "ere ancienne" ; b_h + g_h : reponse "ere recente".

Sorties :
    figures/fig_lp.png       : IRF LP plein echantillon + comparaison eres
    data/lp_results.csv
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

H = 18
L = 6                       # retards de controle
LATE_FROM = 2007            # debut de "l'ere recente"
SAMPLE0 = "1987-05-01"

df = pd.read_csv(DATA / "dataset_monthly.csv", parse_dates=["date"], index_col="date")
d = df.loc[SAMPLE0:, ["temp_shock", "infl", "dloil", "dy10"]].dropna().copy()
d["late"] = (d.index.year >= LATE_FROM).astype(float)
d["shock_late"] = d["temp_shock"] * d["late"]
print("Echantillon LP :", d.index.min().date(), "->", d.index.max().date(), "| n =", len(d))

# matrice de controles : L retards de chaque variable
ctrl_cols = []
for v in ["infl", "temp_shock", "dloil", "dy10"]:
    for l in range(1, L + 1):
        name = f"{v}_l{l}"
        d[name] = d[v].shift(l)
        ctrl_cols.append(name)

rows_full, rows_early, rows_late = [], [], []
for h in range(H + 1):
    d[f"y{h}"] = d["infl"].shift(-h)
    # --- modele plein echantillon (sans interaction) ---
    X = sm.add_constant(d[["temp_shock"] + ctrl_cols])
    reg = sm.OLS(d[f"y{h}"], X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": h + 1})
    b = reg.params["temp_shock"]; se = reg.bse["temp_shock"]
    rows_full.append({"h": h, "b": b, "lo": b - 1.645 * se, "hi": b + 1.645 * se})

    # --- modele avec interaction ere recente ---
    Xi = sm.add_constant(d[["temp_shock", "shock_late", "late"] + ctrl_cols])
    regi = sm.OLS(d[f"y{h}"], Xi, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": h + 1})
    be = regi.params["temp_shock"]; see = regi.bse["temp_shock"]
    gl = regi.params["shock_late"]
    # combinaison lineaire pour l'ere recente + son e.t.
    bl = be + gl
    # variance de (b + g) via matrice de covariance
    cov = regi.cov_params()
    var_bl = (cov.loc["temp_shock", "temp_shock"]
              + cov.loc["shock_late", "shock_late"]
              + 2 * cov.loc["temp_shock", "shock_late"])
    sel = np.sqrt(var_bl)
    rows_early.append({"h": h, "b": be, "lo": be - 1.645 * see, "hi": be + 1.645 * see})
    rows_late.append({"h": h, "b": bl, "lo": bl - 1.645 * sel, "hi": bl + 1.645 * sel,
                      "g": gl, "g_t": gl / regi.bse["shock_late"]})

full = pd.DataFrame(rows_full)
early = pd.DataFrame(rows_early)
late = pd.DataFrame(rows_late)

# Rescalage : exprimer les reponses pour un choc de +1 ecart-type (comme le SVAR)
SD = d["temp_shock"].std()
for tab in (full, early, late):
    for c in ["b", "lo", "hi"]:
        tab[c] *= SD
late["g"] *= SD
print(f"[Rescalage par sd(temp_shock) = {SD:.3f} deg C]")
full.to_csv(DATA / "lp_results.csv", index=False)

print("\n--- IRF LP plein echantillon (reponse de l'inflation au choc temp.) ---")
print(full.round(3).to_string(index=False))
print("\n--- Coefficient d'interaction 'ere recente' (g_h) et t-stat ---")
print(late[["h", "g", "g_t"]].round(3).to_string(index=False))

# ======================================================================
# FIGURE
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
# (a) plein echantillon
ax = axes[0]
ax.fill_between(full["h"], full["lo"], full["hi"], color="#d62728", alpha=0.18)
ax.plot(full["h"], full["b"], color="#d62728", lw=1.9, marker="o", ms=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_title("LP plein echantillon (1987-2026)")
ax.set_xlabel("Mois apres le choc"); ax.set_ylabel("Reponse inflation (pp ann.)")

# (b) comparaison eres
ax = axes[1]
ax.fill_between(early["h"], early["lo"], early["hi"], color="#1f77b4", alpha=0.12)
ax.plot(early["h"], early["b"], color="#1f77b4", lw=1.9, marker="o", ms=3,
        label=f"Ere ancienne (<{LATE_FROM})")
ax.fill_between(late["h"], late["lo"], late["hi"], color="#d62728", alpha=0.12)
ax.plot(late["h"], late["b"], color="#d62728", lw=1.9, marker="s", ms=3,
        label=f"Ere recente (>={LATE_FROM})")
ax.axhline(0, color="k", lw=0.6)
ax.set_title("Intensification : ancienne vs recente")
ax.set_xlabel("Mois apres le choc"); ax.legend(fontsize=8)
fig.suptitle("Local Projections : reponse de l'inflation a un choc de temperature de +1 e.t.",
             fontsize=10.5, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_lp.png", bbox_inches="tight")
plt.close(fig)
print("\nFigure LP enregistree.")
