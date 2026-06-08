"""
06_channels.py  --  ENRICHISSEMENT 2 : les trois canaux de Schnabel (2022)
--------------------------------------------------------------------------
Schnabel distingue climateflation (alea physique), fossilflation (dependance
fossile) et greenflation (transition). On compare les deux premiers, mesurables :
  * climateflation  : choc de temperature
  * fossilflation   : choc de prix du petrole (Brent)  [+ gaz naturel en robustesse]
en confrontant, dans le MEME SVAR, la reponse de l'inflation a chacun et la part de
variance qu'ils expliquent. Objectif : situer l'ampleur RELATIVE du canal climatique.

Sortie : figures/fig_channels.png ; data/channels_results.txt
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; FIG = ROOT / "figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

VARS = ["temp_shock", "dloil", "act", "infl", "dy10"]
HORIZON = 24
df = pd.read_csv(DATA / "dataset_monthly.csv", parse_dates=["date"], index_col="date")
sub = df.loc["1987-05-01":"2015-04-01", VARS].dropna()
res = VAR(sub).fit(2)
irf = res.irf(HORIZON)
fevd = res.fevd(HORIZON)

i_temp, i_oil, i_infl = VARS.index("temp_shock"), VARS.index("dloil"), VARS.index("infl")
# reponses de l'inflation (orthogonalisees, donc deja en +1 e.t. de chaque choc)
resp_temp = irf.orth_irfs[:, i_infl, i_temp]
resp_oil = irf.orth_irfs[:, i_infl, i_oil]
fevd_infl = fevd.decomp[i_infl]      # (h, k)

log = ["=== Climateflation vs fossilflation (SVAR 1987-2015) ===\n",
       "Reponse de l'inflation a un choc de +1 e.t. (pp annualisees) :",
       f"{'h':>3}{'climat (temp)':>16}{'fossile (oil)':>16}"]
for h in [0, 1, 3, 6, 12]:
    log.append(f"{h:3d}{resp_temp[h]:16.3f}{resp_oil[h]:16.3f}")
log.append("\nPart de la variance de l'inflation expliquee (%) :")
log.append(f"{'h':>3}{'temp':>8}{'oil':>8}{'activite':>10}{'infl(propre)':>14}{'taux':>8}")
for h in [6, 12, 24]:
    row = fevd_infl[h - 1] * 100
    log.append(f"{h:3d}{row[i_temp]:8.2f}{row[i_oil]:8.2f}{row[VARS.index('act')]:10.2f}"
               f"{row[i_infl]:14.2f}{row[VARS.index('dy10')]:8.2f}")
ratio = (fevd_infl[23, i_oil]) / (fevd_infl[23, i_temp] + 1e-9)
log.append(f"\nA 24 mois, le canal fossile explique ~{ratio:.0f}x plus de variance "
           f"d'inflation que le canal climatique.")
txt = "\n".join(log)
(DATA / "channels_results.txt").write_text(txt, encoding="utf-8")
print(txt)

# figure
hs = np.arange(HORIZON + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
ax = axes[0]
ax.plot(hs, resp_temp, color="#2ca02c", lw=2.0, marker="o", ms=3, label="Climateflation (temp.)")
ax.plot(hs, resp_oil, color="#8c564b", lw=2.0, marker="s", ms=3, label="Fossilflation (petrole)")
ax.axhline(0, color="k", lw=0.6); ax.set_title("(A) Reponse de l'inflation")
ax.set_xlabel("Mois apres le choc"); ax.set_ylabel("pp annualisees"); ax.legend(fontsize=8)

ax = axes[1]
hsel = [6, 12, 24]
labels = ["Temp.\n(climat)", "Petrole\n(fossile)", "Activite", "Inflation\n(propre)", "Taux"]
idx = [i_temp, i_oil, VARS.index("act"), i_infl, VARS.index("dy10")]
xw = np.arange(len(idx)); w = 0.25
for j, h in enumerate(hsel):
    vals = fevd_infl[h - 1][idx] * 100
    ax.bar(xw + (j - 1) * w, vals, w, label=f"h={h}")
ax.set_xticks(xw); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("% variance inflation"); ax.set_title("(B) Decomposition de variance")
ax.legend(fontsize=8)
fig.suptitle("Canaux d'inflation climatique : climateflation vs fossilflation",
             fontsize=10.5, y=1.04)
fig.tight_layout(); fig.savefig(FIG / "fig_channels.png", bbox_inches="tight")
plt.close(fig)
print("\nFigure canaux enregistree.")
