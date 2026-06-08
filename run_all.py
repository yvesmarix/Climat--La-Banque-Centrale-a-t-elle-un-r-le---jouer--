"""
run_all.py — Exécute l'intégralité du pipeline dans l'ordre.
Usage : python run_all.py
"""
import subprocess, sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"
SCRIPTS = [
    "01_build_dataset.py",      # construction de la base
    "02_svar.py",               # SVAR : croissance + inflation
    "03_local_projections.py",  # LP + intensification
    "04_asset_allocation.py",   # allocation d'actifs
    "05_nonlinear.py",          # ENRICHISSEMENT : non-linearites (asymetrie)
    "06_channels.py",           # ENRICHISSEMENT : climateflation vs fossilflation
    "07_robustness.py",         # ENRICHISSEMENT : robustesse
]
for s in SCRIPTS:
    print(f"\n{'='*60}\n>>> {s}\n{'='*60}")
    r = subprocess.run([sys.executable, str(CODE / s)])
    if r.returncode != 0:
        print(f"ERREUR dans {s}"); sys.exit(r.returncode)
print("\nPipeline terminé. Figures dans figures/, résultats dans data/.")
