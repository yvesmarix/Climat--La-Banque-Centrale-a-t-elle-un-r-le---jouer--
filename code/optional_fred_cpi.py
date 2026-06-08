"""
optional_fred_cpi.py  --  EXTENSION PROPOSEE (a lancer sur votre machine)
-------------------------------------------------------------------------
La litterature (Faccia et al. 2021 ; Kotz et al. 2024) montre que l'effet des chocs
climatiques sur les prix est CONCENTRE sur l'ALIMENTATION (et l'energie). Notre rapport
utilise l'IPC global (signal dilue). Pour tester le mecanisme directement, recuperez les
sous-indices via FRED (acces libre, mais bloque depuis l'environnement de rendu) :

  CPIUFDSL  : CPI - Food (US, mensuel, SA)
  CPIENGSL  : CPI - Energy
  CPILFESL  : CPI - All items less food and energy (inflation sous-jacente)
  CPIAUCSL  : CPI - All items (reference)

Aucune cle API n'est requise pour le telechargement CSV direct de FRED.

Une fois telecharges, vous pouvez relancer 02_svar.py / 03_local_projections.py en
remplacant la variable 'infl' par l'inflation alimentaire, et comparer : on s'attend a
un effet du choc de temperature NETTEMENT plus marque et positif sur l'alimentation que
sur l'IPC global.
"""

import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SERIES = {"food": "CPIUFDSL", "energy": "CPIENGSL",
          "core": "CPILFESL", "headline": "CPIAUCSL"}

def fred_csv(code):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    s = pd.read_csv(url, parse_dates=[0], index_col=0)
    s.columns = [code]
    return s

if __name__ == "__main__":
    frames = []
    for name, code in SERIES.items():
        try:
            s = fred_csv(code); s.columns = [name]
            frames.append(s)
            print(f"OK  {name:9s} ({code}) : {len(s)} obs, {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:
            print(f"ECHEC {name} ({code}) : {e}")
    if frames:
        out = pd.concat(frames, axis=1)
        out.to_csv(DATA / "cpi_components.csv")
        print(f"\nEnregistre : {DATA/'cpi_components.csv'}")
        print("Construisez l'inflation alimentaire : 1200*dlog(food), puis relancez les LP.")
