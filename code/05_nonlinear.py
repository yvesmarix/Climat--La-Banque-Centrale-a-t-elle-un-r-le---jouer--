"""
05_nonlinear.py  --  ENRICHISSEMENT 1 : non-linearites
------------------------------------------------------
La litterature (Kotz et al. 2024) insiste sur le caractere NON LINEAIRE des effets
climatiques : ils sont plus forts pour les chaleurs extremes et concentres en ete.
On teste deux non-linearites par Local Projections DEPENDANTES DE L'ETAT
(a la Ramey & Zubairy 2018) :

(A) Asymetrie chaud / froid :
    infl_{t+h} = c + b_w (I_chaud * choc) + b_f (I_froid * choc) + d*I_chaud + controles
    -> b_w et b_f = pentes de reponse selon que le mois est plus chaud / plus froid
       que la moyenne de l'echantillon.

(B) Saisonnalite ete / hors-ete :
    infl_{t+h} = c + b_ete (ete * choc) + b_hors ((1-ete) * choc) + d*ete + controles

Le choc est recentre sur la moyenne de l'echantillon (un choc positif = mois plus
chaud que la normale de la periode). Erreurs-types HAC (Newey-West). Reponses
exprimees pour un choc de +1 ecart-type.

Sortie : figures/fig_nonlinear.png ; data/nonlinear_results.csv
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
DATA = ROOT / "data"; FIG = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

H, L = 18, 6
df = pd.read_csv(DATA / "dataset_monthly.csv", parse_dates=["date"], index_col="date")
d = df.loc["1987-05-01":, ["temp_shock", "infl", "dloil", "dy10"]].dropna().copy()

# Recentrage local du choc + etats
SD = d["temp_shock"].std()
d["ts"] = d["temp_shock"] - d["temp_shock"].mean()
d["warm"] = (d["ts"] > 0).astype(float)
d["cold"] = 1.0 - d["warm"]
d["summer"] = d.index.month.isin([6, 7, 8]).astype(float)
d["ts_warm"] = d["ts"] * d["warm"]
d["ts_cold"] = d["ts"] * d["cold"]
d["ts_sum"] = d["ts"] * d["summer"]
d["ts_oth"] = d["ts"] * (1 - d["summer"])

ctrl = []
for v in ["infl", "temp_shock", "dloil", "dy10"]:
    for l in range(1, L + 1):
        nm = f"{v}_l{l}"; d[nm] = d[v].shift(l); ctrl.append(nm)

def lp(yvars, regs, dummies):
    """Renvoie {nom: (coef, se)} par horizon pour chaque regresseur d'interet."""
    out = {r: {"b": [], "lo": [], "hi": []} for r in regs}
    for h in range(H + 1):
        d[f"y{h}"] = d["infl"].shift(-h)
        X = sm.add_constant(d[regs + dummies + ctrl])
        m = sm.OLS(d[f"y{h}"], X, missing="drop").fit(cov_type="HAC",
                                                       cov_kwds={"maxlags": h + 1})
        for r in regs:
            b = m.params[r] * SD; se = m.bse[r] * SD
            out[r]["b"].append(b); out[r]["lo"].append(b - 1.645 * se)
            out[r]["hi"].append(b + 1.645 * se)
    return out

asym = lp("y", ["ts_warm", "ts_cold"], ["warm"])

# sauvegarde
hs = np.arange(H + 1)
warm_b = np.array(asym["ts_warm"]["b"]); cold_b = np.array(asym["ts_cold"]["b"])
res = pd.DataFrame({"h": hs, "warm": warm_b, "cold": cold_b,
                    "warm_cum": warm_b.cumsum(), "cold_cum": cold_b.cumsum()})
res.to_csv(DATA / "nonlinear_results.csv", index=False)
print("Reponse de l'inflation par etat (par +1 e.t.) :")
print(res.round(3).to_string(index=False))

# figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
ax = axes[0]
ax.fill_between(hs, asym["ts_warm"]["lo"], asym["ts_warm"]["hi"], color="#d62728", alpha=0.13)
ax.plot(hs, warm_b, color="#d62728", lw=1.9, marker="o", ms=3, label="Mois chaud (> normale)")
ax.fill_between(hs, asym["ts_cold"]["lo"], asym["ts_cold"]["hi"], color="#1f77b4", alpha=0.13)
ax.plot(hs, cold_b, color="#1f77b4", lw=1.9, marker="s", ms=3, label="Mois froid (< normale)")
ax.axhline(0, color="k", lw=0.6); ax.set_title("(A) Reponse par horizon")
ax.set_xlabel("Mois apres le choc"); ax.set_ylabel("Reponse inflation (pp ann.)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(hs, warm_b.cumsum(), color="#d62728", lw=2.0, marker="o", ms=3, label="Mois chaud")
ax.plot(hs, cold_b.cumsum(), color="#1f77b4", lw=2.0, marker="s", ms=3, label="Mois froid")
ax.axhline(0, color="k", lw=0.6); ax.set_title("(B) Reponse cumulee (niveau de prix)")
ax.set_xlabel("Mois apres le choc"); ax.set_ylabel("pp cumulees")
ax.legend(fontsize=8)

fig.suptitle("Asymetrie de la reponse de l'inflation au choc de temperature (LP dependantes de l'etat)",
             fontsize=10.5, y=1.04)
fig.tight_layout(); fig.savefig(FIG / "fig_nonlinear.png", bbox_inches="tight")
plt.close(fig)
print("\nFigure non-linearites enregistree.")
