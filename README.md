# Climat et Banque Centrale — *La climateflation* (SVAR, Local Projections, allocation d'actifs)

Projet d'économétrie — Université Paris-Dauphine, Master Finance & Économétrie
(cours de Politique Monétaire). Réponse à la question **« Le climat : la Banque Centrale
a-t-elle un rôle à jouer ? »** par l'analyse empirique de la transmission des chocs
climatiques à l'inflation, à la croissance et aux prix d'actifs.

## Contenu

```
projet/
├── report/
│   ├── rapport.tex          # rapport LaTeX (source)
│   └── rapport.pdf          # rapport compilé (12 p., ≤ 15)  <-- LIVRABLE PRINCIPAL
├── code/
│   ├── 01_build_dataset.py        # fusion des CSV bruts -> dataset_monthly.csv
│   ├── 02_svar.py                 # SVAR récursif : IRF, FEVD (croissance + inflation)
│   ├── 03_local_projections.py    # Local Projections + test d'intensification
│   ├── 04_asset_allocation.py     # bêtas climatiques + portefeuilles optimaux
│   ├── 05_nonlinear.py            # [ENRICH.] non-linéarité : asymétrie chaud/froid
│   ├── 06_channels.py             # [ENRICH.] climateflation vs fossilflation (Schnabel)
│   ├── 07_robustness.py           # [ENRICH.] robustesse (ordre, retards, détrendage)
│   └── optional_fred_cpi.py       # [EXTENSION] IPC désagrégé alim./énergie via FRED
├── data/
│   ├── *.csv                # données brutes + dataset_monthly.csv (généré)
│   ├── SOURCES.md           # provenance et liens de chaque série
│   └── *_results.txt, *.csv # résultats chiffrés de chaque étape
├── figures/                 # figures PNG (générées par les scripts)
├── requirements.txt
└── run_all.py               # exécute tout le pipeline (01 -> 07)
```

## Reproduire les résultats

```bash
pip install -r requirements.txt
python run_all.py            # exécute 01 à 07 dans l'ordre
```

Ou script par script (`python code/01_build_dataset.py`, etc.). Les données brutes sont
déjà incluses dans `data/` ; pour les retélécharger, voir `data/SOURCES.md`.

## Recompiler le rapport

```bash
cd report && pdflatex rapport.tex && pdflatex rapport.tex   # deux passes (TOC + références)
```

## Démarche (6 étapes)

1. **Choc climatique** : anomalie de température globale (NOAA) détrendée → choc stationnaire.
2. **SVAR récursif** (Cholesky, température exogène ordonnée en 1er) → effet sur **croissance** et **inflation**.
3. **Canaux** : climateflation (température) vs **fossilflation** (pétrole) dans le même SVAR.
4. **Local Projections** (Jordà, HAC) + interaction « ère récente » → **intensification**.
5. **Non-linéarité** : LP dépendantes de l'état → **asymétrie** chaud/froid.
6. **Allocation d'actifs** : bêtas climatiques + portefeuille « climat-neutre ».
   Plus un bloc de **robustesse**.

## Principaux résultats

- Effet agrégé historique **faible** (< 3 % de la variance de l'inflation), activité quasi nulle.
- **Fossilflation ≈ 13× la climateflation** en part de variance d'inflation (≈ 36 % vs 2,7 %).
- Transmission climat→inflation **en hausse depuis 2007** (significative à 14–18 mois).
- Réponse **asymétrique** : chocs chauds inflationnistes, chocs froids désinflationnistes.
- **Obligations** = actif le plus exposé ; **or** = couverture. Neutraliser le risque
  climatique coûte ≈ 14 % de ratio de Sharpe.
- Résultats **robustes** à l'ordre de Cholesky, aux retards et au détrendage.
- Conclusion : rôle de la banque centrale surtout **prospectif et prudentiel**.

> Données 100 % réelles et publiques. Aucune valeur n'est simulée.
