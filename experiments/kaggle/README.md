# Comparaison avec une architecture Kaggle

Adaptation du modèle communautaire de [mrkmakr : AE pretrain + GNN + Attn + CNN](https://www.kaggle.com/code/mrkmakr/covid-ae-pretrain-gnn-attn-cnn).
Le code de départ est le port de ce notebook ajouté par LeBleuH dans le commit `fae06591a86f8e5941e940dd2ccbacb0e3d67d76`, sous le nom `covid-ae-pretrain-gnn-attn-cnn.ipynb`. Le numéro de version du notebook Kaggle n'était pas renseigné dans cet ajout ; nous ne revendiquons donc pas la reproduction d'une version précise ni de son score au leaderboard.

Nous reprenons son architecture : préentraînement d'un autoencodeur débruiteur, convolutions à plusieurs tailles de noyau, agrégation des voisins sur le graphe et attention à quatre têtes. Le code adapté se trouve dans [`src/kaggle/model.py`](../../src/kaggle/model.py), et l'entraînement dans [`experiments/kaggle/train.py`](train.py). Il ne s'agit pas de l'ensemble final du gagnant.

Cette expérience est une **adaptation au protocole commun du projet**, avec les changements suivants par rapport au notebook importé :

| Élément | Notebook importé | Cette expérience |
| --- | --- | --- |
| Filtre des ARN | `signal_to_noise > 1` | `SN_filter = 1` |
| Folds | Nouveau KFold sur les données filtrées | Folds du dépôt, alignés par identifiant |
| Préentraînement | Entrées de tout le train et du test | Entrées du train de chaque fold uniquement |
| Score rapporté | Moyenne sur cinq cibles | Trois cibles officielles |
| Entraînement supervisé | 56 époques, lots de 8 à 256, derniers poids | Maximum 40 époques, lots de 32, meilleur checkpoint |
| Reproductibilité | Seed du découpage uniquement | Seed d'entraînement 42 et vérification des folds |

Les appels Keras ont également été adaptés à Keras 3. Les résultats doivent être comparés à ceux de nos modèles sur les mêmes folds ; ils ne constituent pas une mesure du score Kaggle original.

Pour comparer cette architecture à nos modèles, nous utilisons les mêmes ARN (`SN_filter = 1`) et les affectations exactes de `folds/folds_kfold.csv`, vérifiées par identifiant. Aucun nouveau découpage n'est créé.

L'autoencodeur est réentraîné séparément pour chaque fold, uniquement sur les entrées de son ensemble d'entraînement. Les entrées de validation et de test ne sont jamais utilisées pour le préentraînement. Cette restriction diffère du préentraînement transductif du notebook Kaggle.

Le modèle conserve ses 39 caractéristiques par nucléotide, ses cinq canaux d'adjacence (BPPS, appariements et trois distances), ses convolutions et son attention. Le vocabulaire est fixe. Il n'utilise pas exactement les mêmes caractéristiques que nos architectures : on compare ici des pipelines, pas uniquement un type de couche.

Le score est le MCRMSE global sur les 68 premières positions et les trois cibles officielles : reactivity, deg_Mg_pH10 et deg_Mg_50C. L'apprentissage supervisé conserve les cinq cibles.

Le protocole utilise la seed 42, Adam à 0,001, des lots de 32, au maximum 40 époques supervisées, un arrêt anticipé après huit époques sans amélioration et une réduction du taux après quatre. Le meilleur checkpoint est sélectionné sur le MCRMSE officiel. L'autoencodeur ajoute 20 époques par fold : le budget total et le nombre de paramètres diffèrent donc de ceux des autres modèles.

```bash
.venv/bin/python -m experiments.kaggle.train
```

Les historiques, checkpoints, identifiants de train/validation et prédictions OOF sont dans `outputs/cv_kaggle_ae_common/` (ignoré par Git). Une exécution interrompue peut être relancée avec la même commande : seuls les folds entièrement terminés sont réutilisés. Les scores partiels ne constituent pas un résultat à cinq folds.

Test fonctionnel sur huit ARN, sans valeur scientifique :

```bash
.venv/bin/python -m experiments.kaggle.train --smoke --ae-epochs 1 --epochs 1 --only-fold 0 --name smoke_kaggle_ae
```

Les anciens fichiers OOF et de soumission ne sont plus publiés dans ce dossier. Ils ont été conservés localement dans `outputs/previous_kaggle_upload/` ; les anciens scores ne sont pas mélangés aux nouveaux.

Après cinq folds terminés, le script vérifie les scores à partir des prédictions OOF et enregistre `verification.json` dans le dossier de résultats. Il produit aussi `figures/kaggle/cv_comparison.png`, comparant cette adaptation à la baseline BiGRU sur les mêmes folds. La vérification peut être relancée :

```bash
.venv/bin/python -m experiments.kaggle.verify
```


## Résultat obtenu

Sur les cinq folds communs, la MCRMSE moyenne est **0,22657 ± 0,00539** (écart-type entre folds). Le score calculé en regroupant les prédictions OOF est **0,22663**. La baseline BiGRU annot obtient **0,24933 ± 0,00553**. Cette comparaison concerne deux pipelines : leurs caractéristiques, nombres de paramètres et budgets de préentraînement diffèrent.
