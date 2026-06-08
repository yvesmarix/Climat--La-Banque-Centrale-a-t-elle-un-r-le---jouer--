"""
04_asset_allocation.py
----------------------
Troisieme pilier : implications pour l'ALLOCATION D'ACTIFS.

Etape 1 - Exposition climatique des actifs : on regresse les rendements reels
mensuels de chaque classe d'actifs (actions, obligations 10 ans, or) sur le choc
de temperature (avec controles), pour estimer un "beta climatique".

Etape 2 - Optimisation moyenne-variance :
  * Portefeuille tangent (max Sharpe)        : w ~ Sigma^{-1} mu
  * Portefeuille variance minimale            : w ~ Sigma^{-1} 1
  * Portefeuille "climat-neutre"              : variance min. s.c. somme(w)=1
                                                ET exposition climatique w'beta = 0
On compare les poids et le cout (en Sharpe) de la neutralisation du risque climat.

Sorties :
    figures/fig_assets.png   : betas climatiques + poids des portefeuilles
    data/asset_results.txt
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

ASSETS = ["ret_eq", "ret_bond", "ret_gold"]
ANAMES = {"ret_eq": "Actions", "ret_bond": "Obligations 10a", "ret_gold": "Or"}
SAMPLE0 = "1970-01-01"
L = 3

df = pd.read_csv(DATA / "dataset_monthly.csv", parse_dates=["date"], index_col="date")
d = df.loc[SAMPLE0:, ASSETS + ["temp_shock", "dloil", "infl"]].dropna().copy()
print("Echantillon allocation :", d.index.min().date(), "->", d.index.max().date(),
      "| n =", len(d))

log = ["=== Allocation d'actifs et risque climatique ===",
       f"Echantillon : {d.index.min().date()} -> {d.index.max().date()} (n={len(d)})\n"]

# ----------------------------------------------------------------------
# Etape 1 : betas climatiques (regression avec controles + HAC)
# ----------------------------------------------------------------------
SD = d["temp_shock"].std()
ctrl = []
for v in ["dloil", "infl"]:
    for l in range(1, L + 1):
        nm = f"{v}_l{l}"; d[nm] = d[v].shift(l); ctrl.append(nm)
d["ts_l1"] = d["temp_shock"].shift(1); ctrl.append("ts_l1")

betas, betas_se = {}, {}
log.append("--- Beta climatique des actifs (reponse a un choc de +1 e.t.) ---")
log.append(f"{'Actif':16s}{'beta (pp/mois)':>16s}{'t-stat':>9s}")
for a in ASSETS:
    X = sm.add_constant(d[["temp_shock"] + ctrl])
    r = sm.OLS(d[a], X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    b = r.params["temp_shock"] * SD          # par +1 e.t.
    se = r.bse["temp_shock"] * SD
    betas[a] = b; betas_se[a] = se
    log.append(f"{ANAMES[a]:16s}{b:16.3f}{b/se:9.2f}")
log.append("")

# ----------------------------------------------------------------------
# Etape 2 : optimisation moyenne-variance (rendements reels, rf ~ 0)
# ----------------------------------------------------------------------
R = d[ASSETS].dropna()
mu = R.mean().values * 12.0                  # annualise
Sig = R.cov().values * 12.0
beta = np.array([betas[a] for a in ASSETS])
ones = np.ones(len(ASSETS))
Si = np.linalg.inv(Sig)

def stats(w):
    m = w @ mu; v = w @ Sig @ w
    return m, np.sqrt(v), m / np.sqrt(v), w @ beta

# Portefeuille tangent (max Sharpe, rf=0) : w ~ Sig^{-1} mu
w_tan = Si @ mu; w_tan /= w_tan.sum()
# Variance minimale : w ~ Sig^{-1} 1
w_mv = Si @ ones; w_mv /= w_mv.sum()
# Climat-neutre : min w'Sig w  s.c.  w'1 = 1, w'beta = 0  (Lagrangien, forme close)
A = np.array([[ones @ Si @ ones, ones @ Si @ beta],
              [beta @ Si @ ones, beta @ Si @ beta]])
rhs = np.array([1.0, 0.0])
lam = np.linalg.solve(A, rhs)
w_cn = Si @ (lam[0] * ones + lam[1] * beta)

log.append("--- Portefeuilles (poids %) ---")
hdr = f"{'':18s}" + "".join(f"{ANAMES[a]:>14s}" for a in ASSETS) + \
      f"{'Rdt %':>9s}{'Vol %':>8s}{'Sharpe':>8s}{'beta clim':>11s}"
log.append(hdr)
for name, w in [("Tangent (max Sharpe)", w_tan),
                ("Variance min.", w_mv),
                ("Climat-neutre", w_cn)]:
    m, s, sh, bc = stats(w)
    row = f"{name:18s}" + "".join(f"{100*wi:14.1f}" for wi in w) + \
          f"{m:9.2f}{s:8.2f}{sh:8.3f}{bc:11.3f}"
    log.append(row)

sh_tan = stats(w_tan)[2]; sh_cn = stats(w_cn)[2]
log.append(f"\nCout de la neutralisation climatique : Sharpe {sh_tan:.3f} -> {sh_cn:.3f} "
           f"({100*(sh_cn-sh_tan)/sh_tan:+.1f}%)")
(DATA / "asset_results.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))

# ======================================================================
# FIGURE
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
# (a) betas climatiques
ax = axes[0]
xs = np.arange(len(ASSETS))
bvals = [betas[a] for a in ASSETS]
errs = [1.645 * betas_se[a] for a in ASSETS]
colors = ["#1f77b4", "#d62728", "#bf9000"]
ax.bar(xs, bvals, yerr=errs, color=colors, alpha=0.85, capsize=4)
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels([ANAMES[a] for a in ASSETS])
ax.set_ylabel("Beta climatique (pp/mois, +1 e.t.)")
ax.set_title("Exposition au choc de temperature")
# (b) poids des portefeuilles
ax = axes[1]
width = 0.26
ax.bar(xs - width, 100 * w_tan, width, label="Tangent", color="#2b5d8a")
ax.bar(xs, 100 * w_mv, width, label="Var. min.", color="#7f7f7f")
ax.bar(xs + width, 100 * w_cn, width, label="Climat-neutre", color="#2ca02c")
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels([ANAMES[a] for a in ASSETS])
ax.set_ylabel("Poids (%)"); ax.set_title("Allocation optimale")
ax.legend(fontsize=8)
fig.suptitle(f"Risque climatique et allocation d'actifs (rendements reels, "
             f"{d.index.min().year}-{d.index.max().year})",
             fontsize=10.5, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_assets.png", bbox_inches="tight")
plt.close(fig)
print("\nFigure allocation enregistree.")
