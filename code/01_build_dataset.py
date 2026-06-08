"""
01_build_dataset.py
--------------------
Construit la base de donnees mensuelle fusionnee a partir des fichiers bruts
telecharges depuis des sources publiques (voir data/SOURCES.md).

Sortie : data/dataset_monthly.csv  (index = periode mensuelle)

Variables construites :
  temp_anom   : anomalie de temperature globale (GCAG / NOAA), en deg. C
  temp_shock  : anomalie de temperature detrendee (residu d'une tendance lineaire)
                -> "choc climatique" stationnaire : mois plus chaud/froid que la
                   tendance de rechauffement.
  cpi         : indice des prix a la consommation US (niveau)
  infl        : inflation mensuelle annualisee = 1200 * dlog(cpi)
  oil         : prix du Brent (USD/baril), agrege en moyenne mensuelle
  dloil       : variation log mensuelle du Brent (x100)
  pce_real    : consommation reelle (PCE nominal deflate par le CPI)
  act         : croissance de l'activite reelle = 100 * dlog(pce_real)
  unrate      : taux de chomage (unemploy / pop des 16+ approx via interpolation)
  y10         : taux des obligations d'Etat US a 10 ans (%)
  dy10        : variation mensuelle du taux 10 ans (points de %)
  sp500       : indice S&P 500 (Shiller, niveau)
  div         : dividendes S&P 500 (Shiller)
  ret_eq      : rendement total reel mensuel des actions (%) (prix + dividende, deflate)
  ret_bond    : rendement reel mensuel approche des obligations 10 ans (%)
  gold        : prix de l'or (USD/once)
  ret_gold    : rendement reel mensuel de l'or (%)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def to_period(s):
    return pd.PeriodIndex(pd.to_datetime(s), freq="M")


# ----------------------------------------------------------------------
# 1. Temperature (GCAG = NOAA Global Temp), anomalie mensuelle deg C
# ----------------------------------------------------------------------
temp = pd.read_csv(DATA / "temp_raw.csv")
temp = temp[temp["Source"] == "GCAG"].copy()
temp["period"] = pd.PeriodIndex(temp["Year"], freq="M")
temp = temp.set_index("period")["Mean"].rename("temp_anom").sort_index()

# ----------------------------------------------------------------------
# 2. CPI US
# ----------------------------------------------------------------------
cpi = pd.read_csv(DATA / "cpi_raw.csv")
cpi["period"] = to_period(cpi["Date"])
cpi = cpi.set_index("period")["Index"].rename("cpi").sort_index()

# ----------------------------------------------------------------------
# 3. Brent (quotidien -> moyenne mensuelle)
# ----------------------------------------------------------------------
oil = pd.read_csv(DATA / "brent_raw.csv")
oil["period"] = to_period(oil["Date"])
oil = oil.groupby("period")["Price"].mean().rename("oil").sort_index()

# ----------------------------------------------------------------------
# 4. Activite reelle : PCE nominal (ggplot2 economics) deflate par CPI
# ----------------------------------------------------------------------
eco = pd.read_csv(DATA / "economics.csv")
eco["period"] = to_period(eco["date"])
eco = eco.set_index("period").sort_index()
pce = eco["pce"].rename("pce_nom")
unemploy = eco["unemploy"].rename("unemploy")  # milliers
pop = eco["pop"].rename("pop")                  # milliers

# ----------------------------------------------------------------------
# 5. Taux 10 ans US
# ----------------------------------------------------------------------
y10 = pd.read_csv(DATA / "bond10y.csv")
y10["period"] = to_period(y10["Date"])
y10 = y10.set_index("period")["Rate"].rename("y10").sort_index()

# ----------------------------------------------------------------------
# 6. Actions (Shiller) : prix, dividendes
# ----------------------------------------------------------------------
shi = pd.read_csv(DATA / "sp500.csv")
shi["period"] = to_period(shi["Date"])
shi = shi.set_index("period").sort_index()
sp500 = shi["SP500"].rename("sp500")
div = shi["Dividend"].rename("div")

# ----------------------------------------------------------------------
# 7. Or
# ----------------------------------------------------------------------
gold = pd.read_csv(DATA / "gold.csv")
gold["period"] = pd.PeriodIndex(gold["Date"], freq="M")
gold = gold.set_index("period")["Price"].rename("gold").sort_index()

# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------
df = pd.concat([temp, cpi, oil, pce, unemploy, pop, y10, sp500, div, gold], axis=1)
df = df.sort_index()

# --- Transformations ---
# Inflation mensuelle annualisee (%)
df["infl"] = 1200 * np.log(df["cpi"]).diff()

# Choc petrole (variation log mensuelle, %)
df["dloil"] = 100 * np.log(df["oil"]).diff()

# Activite reelle : PCE deflate par CPI, croissance (%)
df["pce_real"] = df["pce_nom"] / df["cpi"]
df["act"] = 100 * np.log(df["pce_real"]).diff()

# Taux de chomage approche (%)
df["unrate"] = 100 * df["unemploy"] / (df["pop"])  # proxy (pop totale) -> tendance OK
# variation du taux 10 ans
df["dy10"] = df["y10"].diff()

# Rendements d'actifs reels mensuels (%)
# Actions : (P_t + D_t/12) / P_{t-1} - 1, puis deflate par inflation mensuelle
infl_m = np.log(df["cpi"]).diff()  # inflation mensuelle (log)
eq_nom = (df["sp500"] + df["div"] / 12.0) / df["sp500"].shift(1) - 1.0
df["ret_eq"] = 100 * (np.log1p(eq_nom) - infl_m)
# Obligations 10 ans : approximation rendement = portage + effet duration
# r_bond ~ y_{t-1}/12 - D*(y_t - y_{t-1}), duration D ~ 8 ans (approx 10y)
D = 8.0
df["ret_bond"] = 100 * ((df["y10"].shift(1) / 100.0) / 12.0
                        - D * (df["y10"] - df["y10"].shift(1)) / 100.0
                        - infl_m)
# Or : rendement reel
df["ret_gold"] = 100 * (np.log(df["gold"]).diff() - infl_m)

# --- Choc de temperature : detrendage lineaire de l'anomalie ---
t = df.dropna(subset=["temp_anom"]).copy()
x = np.arange(len(t))
coef = np.polyfit(x, t["temp_anom"].values, 1)
trend = np.polyval(coef, x)
t["temp_shock"] = t["temp_anom"].values - trend
df = df.join(t["temp_shock"])

# Sauvegarde complete
out = df.copy()
out.index = out.index.to_timestamp()
out.index.name = "date"
out.to_csv(DATA / "dataset_monthly.csv")

print("Base construite :", out.shape)
print("Periode :", out.index.min().date(), "->", out.index.max().date())
print("\nColonnes :", list(out.columns))
print("\nApercu (variables cles, lignes recentes) :")
cols = ["temp_anom", "temp_shock", "infl", "dloil", "act", "y10", "ret_eq", "ret_bond", "ret_gold"]
print(out[cols].dropna(how="all").tail(6).round(3).to_string())
