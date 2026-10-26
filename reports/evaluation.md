# Évaluation de la détection

5,000 clients, dont 150 suspects (vérité terrain).

## Métriques globales

| methode          |   precision@100 |   rappel@100 |   precision@200 |   rappel@200 |   precision@300 |   rappel@300 |
|:-----------------|----------------:|-------------:|----------------:|-------------:|----------------:|-------------:|
| Isolation Forest |            0.63 |         0.42 |            0.43 |         0.57 |            0.32 |         0.63 |
| Règles           |            0.63 |         0.42 |            0.5  |         0.67 |            0.36 |         0.71 |
| Hybride          |            0.63 |         0.42 |            0.52 |         0.69 |            0.39 |         0.78 |

## Rappel par typologie et intensité (top 300)

|                                     |   Isolation Forest |   Règles |   Hybride |
|:------------------------------------|-------------------:|---------:|----------:|
| ('compte_de_passage', 'discrete')   |               0.2  |     0.4  |      0.4  |
| ('compte_de_passage', 'forte')      |               1    |     1    |      1    |
| ('crypto', 'discrete')              |               0.73 |     0.27 |      0.6  |
| ('crypto', 'forte')                 |               1    |     1    |      1    |
| ('fractionnement', 'discrete')      |               0.8  |     0.67 |      0.87 |
| ('fractionnement', 'forte')         |               1    |     1    |      1    |
| ('incoherence_revenus', 'discrete') |               0.4  |     0.07 |      0.33 |
| ('incoherence_revenus', 'forte')    |               1    |     1    |      1    |
| ('pays_a_risque', 'discrete')       |               0.07 |     0.73 |      0.6  |
| ('pays_a_risque', 'forte')          |               0.13 |     1    |      1    |

## Clients légitimes atypiques dans le top 300 (hybride)

| profil                   |   nb_clients |
|:-------------------------|-------------:|
| famille_pays_a_risque    |           49 |
| heritage_reinvesti       |           46 |
| gros_investisseur_crypto |           11 |
