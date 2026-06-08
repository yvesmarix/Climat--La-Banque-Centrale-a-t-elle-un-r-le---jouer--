"""
02_svar.py
----------
SVAR mensuel a identification recursive (Cholesky) du choc climatique.

Ordre causal (du plus exogene au plus endogene) :
    temp_shock -> dloil -> act -> infl -> y10
Hypothese d'identification : un choc de temperature (mois plus chaud/froid que
la tendance) n'est pas cause de maniere contemporaine par l'economie ; il est
donc place en premier (Faccia, Parker & Stracca 2021 ; Kotz et al. 2024).

Sorties :
    figures/fig_series.png        : series brutes
    figures/fig_irf_svar.png      : IRF inflation & activite a un choc de +1 e.t. de temperature
    figures/fig_fevd.png          : decomposition de variance de l'inflation
    data/svar_results.txt         : tests, selection de retards, valeurs propres
    data/irf_svar.csv             : IRF ponctuelles + bornes bootstrap
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

np.random.seed(12345)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

# ----------------------------------------------------------------------
VARS = ["temp_shock", "dloil", "act", "infl", "dy10"]
LABELS = {"temp_shock": "Choc temp. (deg C)", "dloil": "Var. log Brent (%)",
          "act": "Croissance reelle (%)", "infl": "Inflation annualisee (%)",
          "dy10": "Var. taux 10 ans (pp)"}
SAMPLE = ("1987-05-01", "2015-04-01")   # intersection oil + activite reelle
HORIZON = 24
NBOOT = 1000

df = pd.read_csv(DATA / "dataset_monthly.csv", parse_dates=["date"], index_col="date")
sub = df.loc[SAMPLE[0]:SAMPLE[1], VARS].dropna()
print("Echantillon SVAR :", sub.index.min().date(), "->", sub.index.max().date(),
      "| n =", len(sub))

log = []
log.append("=== SVAR climateflation : diagnostics ===")
log.append(f"Echantillon : {sub.index.min().date()} -> {sub.index.max().date()}  (n={len(sub)})\n")

# ---- Tests ADF de stationnarite ----
log.append("--- Tests ADF (H0 : racine unitaire) ---")
for c in VARS:
    stat, p, *_ = adfuller(sub[c].dropna(), autolag="AIC")
    log.append(f"{c:12s} ADF={stat:7.3f}  p={p:6.3f}  {'stationnaire' if p<0.05 else 'NON stat.'}")
log.append("")

# ---- Selection du nombre de retards ----
model = VAR(sub)
sel = model.select_order(maxlags=12)
log.append("--- Selection du nombre de retards ---")
log.append(str(sel.summary()))
p_aic = sel.aic
plag = int(p_aic) if p_aic and p_aic > 0 else 3
log.append(f"\nRetards retenus (AIC) : {plag}\n")

# ---- Estimation ----
res = model.fit(plag)
roots = res.roots                      # racines du polynome ; |.|>1 = stable
modroots = np.abs(roots)
log.append(f"Module des racines (toutes > 1 => stable) : min={modroots.min():.3f}")
log.append(f"Systeme stable : {bool(modroots.min() > 1.0)}\n")

# ---- IRF Cholesky (orthogonalisees) + bandes analytiques 90% ----
irf = res.irf(HORIZON)
ti = VARS.index("temp_shock")
se = irf.stderr(orth=True)          # erreurs-types asymptotiques, forme (h+1,k,k)
z = 1.645                            # quantile 90%
lower = irf.orth_irfs - z * se
upper = irf.orth_irfs + z * se

# Stockage des IRF d'interet : reponse de infl et act au choc temp
records = []
ii_infl = VARS.index("infl")
ii_act = VARS.index("act")
ii_oil = VARS.index("dloil")
for h in range(HORIZON + 1):
    records.append({
        "h": h,
        "infl_resp": irf.orth_irfs[h, ii_infl, ti],
        "infl_lo": lower[h, ii_infl, ti],
        "infl_hi": upper[h, ii_infl, ti],
        "act_resp": irf.orth_irfs[h, ii_act, ti],
        "act_lo": lower[h, ii_act, ti],
        "act_hi": upper[h, ii_act, ti],
        "oil_resp": irf.orth_irfs[h, ii_oil, ti],
    })
irf_df = pd.DataFrame(records)
irf_df["infl_cum"] = irf_df["infl_resp"].cumsum()
irf_df.to_csv(DATA / "irf_svar.csv", index=False)
print("\nProfil IRF inflation (h, reponse, cumul) :")
print(irf_df[["h", "infl_resp", "infl_cum"]].head(13).round(3).to_string(index=False))
log.append("--- IRF (reponse a un choc de +1 e.t. de temperature) ---")
log.append("Reponse pic de l'inflation : "
           f"{irf_df['infl_resp'].abs().max():.3f} pp annualises "
           f"(horizon {irf_df['infl_resp'].abs().idxmax()})")
log.append(f"Reponse cumulee de l'inflation a 12 mois : {irf_df['infl_cum'].iloc[12]:.3f}")
log.append(f"Reponse pic de l'activite : {irf_df['act_resp'].abs().max():.3f} pp\n")

# ---- FEVD ----
fevd = res.fevd(HORIZON)
# part de la variance de l'inflation expliquee par le choc temp
fevd_infl = fevd.decomp[ii_infl]    # (horizon, k)
log.append("--- Decomposition de variance de l'inflation (part du choc temperature) ---")
for h in [1, 6, 12, 24]:
    log.append(f"  h={h:2d} : {100*fevd_infl[h-1, ti]:5.2f}%")

(DATA / "svar_results.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log[:40]))

# ======================================================================
# FIGURES
# ======================================================================
# 1) Series
fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.2))
plot_vars = ["temp_anom", "temp_shock", "infl", "dloil", "act", "y10"]
plabels = {"temp_anom": "Anomalie de temperature (deg C)",
           "y10": "Taux 10 ans (%)", **LABELS}
for ax, c in zip(axes.ravel(), plot_vars):
    s = df.loc["1980":"2026", c].dropna()
    ax.plot(s.index, s.values, lw=0.8, color="#2b5d8a")
    ax.set_title(plabels.get(c, c), fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle("Series mensuelles utilisees (1980-2026)", fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(FIG / "fig_series.png", bbox_inches="tight")
plt.close(fig)

# 2) IRF inflation & activite
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
h = irf_df["h"]
# inflation
axes[0].fill_between(h, irf_df["infl_lo"], irf_df["infl_hi"], color="#d62728", alpha=0.18)
axes[0].plot(h, irf_df["infl_resp"], color="#d62728", lw=1.8)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_title("Reponse de l'inflation")
axes[0].set_ylabel("pp annualises")
# activite
axes[1].fill_between(h, irf_df["act_lo"], irf_df["act_hi"], color="#1f77b4", alpha=0.18)
axes[1].plot(h, irf_df["act_resp"], color="#1f77b4", lw=1.8)
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_title("Reponse de l'activite reelle")
axes[1].set_ylabel("pp")
# inflation cumulee
axes[2].plot(h, irf_df["infl_cum"], color="#9467bd", lw=1.8)
axes[2].axhline(0, color="k", lw=0.6)
axes[2].set_title("Inflation cumulee")
axes[2].set_ylabel("pp (niveau de prix)")
for ax in axes:
    ax.set_xlabel("Mois apres le choc")
fig.suptitle("Reponses a un choc de temperature de +1 ecart-type (bandes 90% asymptotiques)",
             fontsize=10.5, y=1.04)
fig.tight_layout()
fig.savefig(FIG / "fig_irf_svar.png", bbox_inches="tight")
plt.close(fig)

# 3) FEVD de l'inflation (contributions empilees)
fig, ax = plt.subplots(figsize=(7, 3.6))
contrib = fevd_infl[:, :] * 100
hh = np.arange(1, HORIZON + 1)
bottom = np.zeros(HORIZON)
colors = ["#2ca02c", "#8c564b", "#1f77b4", "#d62728", "#7f7f7f"]
for k, c in enumerate(VARS):
    ax.bar(hh, contrib[:, k], bottom=bottom, label=LABELS[c], color=colors[k], width=0.9)
    bottom += contrib[:, k]
ax.set_xlabel("Horizon (mois)")
ax.set_ylabel("% de la variance de l'inflation")
ax.set_title("Decomposition de la variance de l'inflation")
ax.legend(fontsize=7.5, ncol=2, loc="lower right")
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(FIG / "fig_fevd.png", bbox_inches="tight")
plt.close(fig)

print("\nFigures SVAR enregistrees.")
