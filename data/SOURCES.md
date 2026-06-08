# Sources des données

Toutes les séries sont publiques, mensuelles, et ont été téléchargées depuis des
dépôts ouverts (datahub.io et miroirs GitHub). Aucune donnée n'est simulée.

| Fichier brut | Contenu | Source |
|---|---|---|
| `temp_raw.csv` | Anomalie de température globale (GCAG / NOAA), mensuelle depuis 1850 | https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv |
| `cpi_raw.csv` | Indice des prix à la consommation US (IPC), mensuel depuis 1913 | https://raw.githubusercontent.com/datasets/cpi-us/master/data/cpiai.csv |
| `brent_raw.csv` | Prix du pétrole Brent, quotidien depuis 1987 (agrégé en moyenne mensuelle) | https://raw.githubusercontent.com/datasets/oil-prices/master/data/brent-daily.csv |
| `economics.csv` | Consommation (PCE), population, chômage — jeu *economics* de ggplot2, 1967–2015 | https://raw.githubusercontent.com/selva86/datasets/master/economics.csv |
| `sp500.csv` | Base de Robert Shiller : S&P 500, dividendes, bénéfices, IPC, taux long 10 ans, depuis 1871 | https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv |
| `gold.csv` | Prix de l'or (USD/once), mensuel depuis 1833 | https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv |
| `bond10y.csv` | Taux des obligations d'État US à 10 ans, mensuel depuis 1953 | https://raw.githubusercontent.com/datasets/bond-yields-us-10y/master/data/monthly.csv |

Le fichier `dataset_monthly.csv` est **généré** par `code/01_build_dataset.py` à partir
des fichiers bruts ci-dessus (fusion + transformations). Voir le script pour le détail
de la construction de chaque variable.

## Mesure du choc climatique
L'anomalie de température globale présente une forte tendance haussière (réchauffement).
Le **choc** utilisé dans les modèles est le résidu d'une régression de l'anomalie sur une
tendance linéaire : il représente un mois *plus chaud (ou plus froid) que la tendance*, et
il est stationnaire (test ADF : p = 0,007). C'est l'approche standard de la littérature
(Faccia, Parker & Stracca 2021 ; Kotz et al. 2024).
