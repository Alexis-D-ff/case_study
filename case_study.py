# %% [markdown]
# # <a id='toc1_'></a>[Ornikar Case Study](#toc0_)

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [Ornikar Case Study](#toc1_)    
#   - [Case study principal](#toc1_1_)    
#     - [EDA](#toc1_1_1_)    
#       - [Vérifier les doublons](#toc1_1_1_1_)    
#       - [Encoding des cibles](#toc1_1_1_2_)    
#       - [Contrôle du déséquilibre des cibles](#toc1_1_1_3_)    
#     - [Préparation des données](#toc1_1_2_)    
#       - [Data types](#toc1_1_2_1_)    
#       - [Traitement NaNs](#toc1_1_2_2_)    
#       - [Heatmap de corrélation de Spearman](#toc1_1_2_3_)    
#       - [Datetime encoding](#toc1_1_2_4_)    
#       - [Vérifier les features de variance nulle](#toc1_1_2_5_)    
#     - [ML modeling](#toc1_1_3_)    
#       - [Baseline](#toc1_1_3_1_)    
#         - [Scoring metric](#toc1_1_3_1_1_)    
#         - [Ensemble de modèles](#toc1_1_3_1_2_)    
#         - [Encoding des features catégoriels et la validation croisée](#toc1_1_3_1_3_)    
#       - [Fine-tuning](#toc1_1_3_2_)    
#         - [Gérer le déséquilibre des cibles](#toc1_1_3_2_1_)    
#         - [Feature importance](#toc1_1_3_2_2_)    
#     - [Conclusion](#toc1_1_4_)    
#     - [Résultats business](#toc1_1_5_)    
#   - [Deux petits casse-têtes](#toc1_2_)    
#     - [Présence de graphes au sein d’un dataset](#toc1_2_1_)    
#     - [Etudes contradictoires](#toc1_2_2_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %%
# %load_ext autoreload
# %autoreload 2
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## <a id='toc1_1_'></a>[Case study principal](#toc0_)
# Objectif : apprendre à prédire de façon efficace la conversion de nos leads en clients, en un
# temps limité, avec les données disponibles.

# %%
from src import constants

# %%
df_raw = pd.read_csv(pathlib.Path(constants.QUOTES), index_col=0)
df_raw.head(5)

# %% [markdown]
# ### <a id='toc1_1_1_'></a>[EDA](#toc0_)
# #### <a id='toc1_1_1_1_'></a>[Vérifier les doublons](#toc0_)

# %%
duplicates = df_raw[df_raw.duplicated(keep=False)]
if duplicates.empty:
    print("No duplicate rows in the data")
else:
    print(f"Total rows with duplicates: {duplicates.shape[0]}")

# %%
duplicates

# %%
# Keep only the first duplicated row
df = df_raw[~df_raw.duplicated(keep='first')].reset_index(drop=True)

# %% [markdown]
# #### <a id='toc1_1_1_2_'></a>[Encoding des cibles](#toc0_)

# %%
df["has_subscribed"] = df["has_subscribed"].astype(int)

# %% [markdown]
# #### <a id='toc1_1_1_3_'></a>[Contrôle du déséquilibre des cibles](#toc0_)

# %%
pd.concat([df["has_subscribed"].value_counts(normalize=True)*100,
           df["has_subscribed"].value_counts()],
          axis=1,
          keys=('perc','count'))

# %% [markdown]
# ❗ L'ensemble de données contient des targets déséquilibrés. Les techniques de traitement du déséquilibre doivent être appliquées pour la modélisation.

# %% [markdown]
# ### <a id='toc1_1_2_'></a>[Préparation des données](#toc0_)
# #### <a id='toc1_1_2_1_'></a>[Data types](#toc0_)
# 
# Tout d'abord, nous allons définir visuellement les types de données des colonnes.

# %%
df.iloc[:3, :len(df.columns)//2]

# %%
df.iloc[:3, len(df.columns)//2:]

# %%
# According to the visual view
cat_features = ['long_quote_id',
                'lead_id',
                'main_driver_age',
                'main_driver_gender',
                'main_driver_licence_age',
                'main_driver_bonus',
                'vehicle_age',
                'vehicle_class',
                'vehicle_group',
                'vehicle_region',]

bool_features = ['has_been_proposed_formulas',
                 'has_chosen_formula',
                 'has_subscribed_online',
                 'has_secondary_driver',]

datetime_features = ['submitted_at',
                     'effective_start_date',]

# %% [markdown]
# Pour le reste, les colonnes avec moins de 10 valeurs uniques seront définies comme catégorielles.
# 
# **Vérifions les colonnes avec plus de 10 valeurs uniques**

# %%
df_rest = df.drop(cat_features + bool_features + datetime_features, axis=1)
df_rest[df_rest.columns[df_rest.nunique() > 10]]

# %% [markdown]
# ❗ La colonne `'policy_subscribed_at'` est datetime. Toutes les autres sont catégorielles.

# %%
datetime_features.extend(['policy_subscribed_at'])
residual_cat_cols = [col for col in df.columns if col not in (cat_features + bool_features + datetime_features + ['has_subscribed'])]
cat_features.extend(residual_cat_cols)

# %%
from src.data_prep import feature_transform

# %%
df_transformed = feature_transform.set_dtypes(df,
                                              cat_cols=cat_features,
                                              bool_cols=bool_features,
                                              datetime_cols=datetime_features)

# %%
df_transformed.dtypes

# %% [markdown]
# La colonne `'has_subscribed_online'` ressemble à un feature lié au target

# %%
df_transformed[['has_subscribed_online', 'has_subscribed']].value_counts().to_frame(name='counts')

# %% [markdown]
# ❗ Tous les utilisateurs `'has_subscribed_online'` positifs ont souscrit un contrat. Il s'agit d'un feature lié à la cible (ne sera pas disponible pour l'inférence). Il devrait être supprimé.

# %%
df_transformed.drop('has_subscribed_online', axis=1, inplace=True)

# %% [markdown]
# #### <a id='toc1_1_2_2_'></a>[Traitement NaNs](#toc0_)

# %%
df_nans = pd.DataFrame((df.isna().mean()*100).sort_values(ascending=False)).rename(columns={0: '%_NaNs'})

# %%
px.scatter(df_nans,
           x=df_nans.index,
           y="%_NaNs",
           title="Count of NaNs in data",
           ).update_layout(
               xaxis_title="Sensor number",
               yaxis_title="% of NaNs")

# %% [markdown]
# Pipeline de traitement NaN :
# 1. Drop les colonnes 100% NaN
# 2. Remplacez NaN par 'not_provided' pour les colonnes catégorielles
# 3. Transformez la colonne datetime `'policy_subscribed_at'` en binaire (valeur présente/non)
# 4. Imputer la colonne datetime `'effective_start_date'`. Mettre la distance médiane entre `'submitted_at'` et la `'effective_start_date'` du dataset complet.

# %%
# Drop cols with 100% of NaNs
df_transformed.dropna(axis=1, how='all', inplace=True)

# %%
# NaN -> not_provided in cat cols
df_transformed = feature_transform.slice_nan_imputer(df_transformed,
                                                     list_cols=df_transformed.select_dtypes(include=object).columns,
                                                     string='not_provided')

# %%
# Binarize the `policy_subscribed_at` column
df_transformed = feature_transform.col_binarize(df_transformed,
                                                list_cols=['policy_subscribed_at'],)

# %%
# Imputation of the `effective_start_date` column
df_transformed = feature_transform.esd_imputer(df_transformed)

# %%
# Verify the nans
df_transformed.isna().sum().sum()

# %% [markdown]
# #### <a id='toc1_1_2_3_'></a>[Heatmap de corrélation de Spearman](#toc0_)

# %%
corr = df_transformed.apply(lambda x : pd.factorize(x)[0]).corr(method='spearman', min_periods=1) # factorize cat features, use the spearman rank correlation
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, mask=np.triu(corr));

# %%
# Visualize high correlating features
from src.data_viz import viz_funcs

viz_funcs.corr_viz(corr, threshold=0.85)

# %%
df_transformed[['long_quote_id', 'submitted_at']].nunique()

# %% [markdown]
# ❗ Certaines insights business peuvent être déduits. La corrélation entre `'long_quote_id'` et `'submitted_at'` est liée à l'unicité de ces colonnes. Certains modèles ML peuvent être sensibles à la multicolinéarité. Nous appliquerons des algorithmes tolérants à la multicolinéarité et des méthodes de régularisation pour gérer cela.

# %% [markdown]
# #### <a id='toc1_1_2_4_'></a>[Datetime encoding](#toc0_)
# 
# Afin d'utiliser les informations des colonnes datetime, le pipeline suivant pour l'encodage datetime sera appliqué :
# 1. Extraire l'année en tant que valeur int
# 2. Extraire le nom du mois en tant que valeur catégoriel
# 3. Extraire le jour comme valeur int
# 4. Si le temps est fourni, ajouter le sinus et le cosinus du temps comme des nouveaux features.

# %%
df_transformed = feature_transform.datetime_encoding(df_transformed)

# %% [markdown]
# #### <a id='toc1_1_2_5_'></a>[Vérifier les features de variance nulle](#toc0_)

# %%
zero_var_cols = df_transformed.var()[df_transformed.var() == 0].index
zero_var_cols

# %% [markdown]
# ❗ Dropping 0-vav columns

# %%
df_transformed.drop(zero_var_cols, axis=1, inplace=True)

# %% [markdown]
# ### <a id='toc1_1_3_'></a>[ML modeling](#toc0_)
# #### <a id='toc1_1_3_1_'></a>[Baseline](#toc0_)
# ##### <a id='toc1_1_3_1_1_'></a>[Scoring metric](#toc0_)
# 
# Il existe un ensemble de métriques ML pour les problèmes de classification binaire, basées sur les combinaisons des prédictions TP, TN, FN et FP.
#  
# **Precision** calcule le rapport entre les vrais positifs et le total des positifs prédits par un classifieur.
# 
# True positive rate, aka **Recall**, nous donne le nombre de vrais positifs divisé par le nombre total d'éléments qui appartiennent réellement à la classe positive.
# 
# Dans notre cas, le **Recall** est une mesure plus importante que la **Precision** étant donné que nous sommes plus préoccupés par les faux négatifs (notre modèle prédit que quelqu'un ne va pas signer le contrat, mais ils le font) que par les faux positifs (notre modèle prédit que quelqu'un va signer le contrat mais ils ne le font pas).
# 
# Cependant, parfois (et c'est le cas de cette étude) l'utilisation du **Recall** comme métrique d'évaluation pour la validation croisée conduit à un overfitting sur les données train et à une faible généralisation du modèle.
# 
# Pour y faire face, nous allons utiliser le **ROC AUC score** qui prend en compte à la fois le **Recall** et le **Fall-out (False Positive Rate)**. Un modèle avec une capacité de discrimination élevée aura simultanément un TPR et un FPR élevés.
# 
# --------------------
# 
#     ❗ Solution optimale : avec les coûts business estimatifs des prédictions TP, FP et FN, un scoreur d'évaluation personnalisée pourrait être mis en œuvre. Cette métrique personnalisée peut prendre en compte les pertes dans les cas FN et FP et les gains dans les cas TP. En combinant ces estimations, un modèle pourrait optimiser le revenu total en adaptant le seuil de discrimination.
# 
# ##### <a id='toc1_1_3_1_2_'></a>[Ensemble de modèles](#toc0_)
# Les modèles ML suivants vont être testés de base:
# 
#     1. SVM avec noyau polynomial
#     2. SVM avec noyau rbf
#     3. Logistic Regression
#     4. KNeighbors Classifier
#     5. Random Forest Classifier
#     6. LGBM Classifier
# 
# L'idée est de tester des modèles tolérants au réglage fin et de nature différente.
# 
# ##### <a id='toc1_1_3_1_3_'></a>[Encoding des features catégoriels et la validation croisée](#toc0_)
# 
# Nous allons appliquer l'encodeur Helmert contrast aux features catégoriels du dataset actuel. L'encodeur Helmert compare les niveaux d'une variable avec la moyenne des niveaux suivants de cette variable.
# 
# De plus, nous allons utiliser une fonction de validation croisée personnalisée pour la modélisation de base afin d'implémenter l'encoding catégoriel avec une seule étape de validation (les features catégoriels sont encodées sur le dataset test pour chaque split et les scores sont calculés comme moyenne pour chaque modèle).

# %%
from sklearn.metrics import roc_auc_score, recall_score
from src.estimation import baseline_test

# %%
estimators_w_sclng = baseline_test.baseline(df_transformed.drop('has_subscribed', axis=1),
                                   df_transformed['has_subscribed'],
                                   est_list=constants.PP_W_SCLNG,
                                   shuffle=True,
                                   random_state=constants.RND_SEED,
                                   scoring=roc_auc_score)

estimators_wo_sclng = baseline_test.baseline(df_transformed.drop('has_subscribed', axis=1),
                                   df_transformed['has_subscribed'],
                                   est_list=constants.PP_WO_SCLNG,
                                   shuffle=True,
                                   random_state=constants.RND_SEED,
                                   scoring=roc_auc_score)

# %%
estimators_w_sclng, estimators_wo_sclng

# %% [markdown]
# ❗ LGBM gradient booster est notre solution à fine-tune

# %% [markdown]
# #### <a id='toc1_1_3_2_'></a>[Fine-tuning](#toc0_)

# %%
from sklearn.model_selection import train_test_split
from category_encoders.helmert import HelmertEncoder

# %% [markdown]
# ##### <a id='toc1_1_3_2_1_'></a>[Gérer le déséquilibre des cibles](#toc0_)
# 
# Il existe plusieurs méthodes pour gérer le déséquilibre des données :
# 1. Suréchantillonnage de la classe minoritaire
# 2. Sous-échantillonnage de la classe majoritaire
# 3. Pénaliser l'algorithme ML
# 
# Selon les conclusions des résultats de base, nous allons sélectionner le modèle de gradient booster, dans l'implémentation LightGBM de Microsoft.
# Cet algorithme contient une méthode simple pour pénaliser les classes lors de l'entraînement. Ainsi, la 3ème approche de gestion du déséquilibre sera choisie.

# %%
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df_transformed.drop('has_subscribed', axis=1),
                                                    df_transformed['has_subscribed'],
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    stratify=df_transformed['has_subscribed'],
                                                    random_state=constants.RND_SEED)

# %%
cat_features = list(X_train_raw.select_dtypes(include=[object]).columns)
helm_encoder = HelmertEncoder(cols=cat_features)

X_train = helm_encoder.fit_transform(X_train_raw, y_train)
X_test = helm_encoder.transform(X_test_raw)

# %%
from lightgbm import LGBMClassifier
from optuna.integration import OptunaSearchCV

scaling_weight = round(df_transformed['has_subscribed'].value_counts()[0]/df_transformed['has_subscribed'].value_counts()[1])
clf = LGBMClassifier(scale_pos_weight=scaling_weight,
                     random_state=constants.RND_SEED,
                     verbose=-1)

gs = OptunaSearchCV(clf,
                    constants.OPT_HYPER_PARAMS,
                    random_state=constants.RND_SEED,
                    scoring='roc_auc',
                    n_trials=45,
                    timeout=None,)

# %%
gs.fit(X_train, y_train)

# %%
lgbm = gs.predict(X_test)

# %%
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, f1_score

# %%
# Visualize the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, lgbm, cmap='binary', colorbar=False)
plt.grid(False)

# %%
tuned_roc_auc = roc_auc_score(y_test, lgbm)
tuned_recall = recall_score(y_test, lgbm)
tuned_f1 = f1_score(y_test, lgbm)
print(f"The recall test score of the tuned LGBM classifier is: {tuned_recall:.3f}")
print('-------------------------------------------------------------')
print(f"The ROC AUC test score of the tuned LGBM classifier is: {tuned_roc_auc:.3f}")
print(f"The f1 test score of the tuned LGBM classifier is: {tuned_f1:.3f}")

# %% [markdown]
# ##### <a id='toc1_1_3_2_2_'></a>[Feature importance](#toc0_)
# 
# Comme le dataframe d'origine a été encodée avec l'encodeur Helmert, j'ai créé une fonction de traçage personnalisée de l'importance des features, qui additionne l'importance des features encodés pour représenter les colonnes d'origine.

# %%
viz_funcs.custom_feature_imp_plot(gs.best_estimator_,
                                  importance_type='gain',
                                  default_cols=df.columns,
                                  encoded_cols=X_train.columns,
                                  figsize=(12, 6),);

# %% [markdown]
# ### <a id='toc1_1_4_'></a>[Conclusion](#toc0_)
# - *ROC AUC score* est sélectionné comme métrique d'évaluation optimale.
# - Cinq différents types de modèles ML ont été testés pour la prédiction de base.
# - **Helmert** encodeur a été utilisé pour l'encodage des features catégoriels.
# - Chaque modèle de base a été validé avec une fonction cv personnalisée. L'ensemble de test de chaque split contenait ses propres features catégoriels encodés par **Helmert**.
# - Gradient boosted ensembles of trees montre un meilleur résultat par rapport à d'autres modèles.
# - L'optimisation bayésienne des hyperparamètres a permis d'obtenir le *Recall* de **0,591**.
# 
# 
# ### <a id='toc1_1_5_'></a>[Résultats business](#toc0_)
# 
# - Le *Recall* de **0,591** signifie qu'environ 59% des prospects susceptibles de signer un contrat peuvent être prédits par le modèle.
# - Selon feature importance, le feature `effective_start_date` est prédominant en matière de prédiction, 3 autres features importants sont :
# 
#         provider
#         policy_subscribed_at
#         last_utm_source

# %% [markdown]
# ## <a id='toc1_2_'></a>[Deux petits casse-têtes](#toc0_)
# ### <a id='toc1_2_1_'></a>[Présence de graphes au sein d’un dataset](#toc0_)
# Mon idée:
# 1. Ajouter `'family'` cat feature
# 2. Ajoutez `'family_member_canceled'` cat feature
# 2. Le leakadge cible doit être exclu, de sorte que la méthode de validation croisée StratifiedGroupKFold peut être optimale. Chaque groupe ne sera pas divisé entre les sets de test et de train.
# 3. La régularisation est utilisée pour éviter l'over-fitting des modèles. Cela peut augmenter la généralisation, mais ce n'est pas garanti.

# %% [markdown]
# ### <a id='toc1_2_2_'></a>[Etudes contradictoires](#toc0_)

# %% [markdown]
# C'est le [Paradoxe de Simpson](https://en.wikipedia.org/wiki/Simpson%27s_paradox), un phénomène en statistique quand, en présence de deux groupes de données, dans chacun desquels il y a une dépendance également dirigée, lorsque ces groupes sont combinés, le sens de la dépendance est inversé.
# 
# On suppose intuitivement que s'il existe une dépendance dans les deux groupes, elle devrait également apparaître lorsque ces groupes sont combinés. Mais en raison de la non-représentativité du groupe témoin dans les données agrégées, cette tendance ne persiste pas.
# 
# Pour un calcul correct de la moyenne (étude A dans notre cas), il faut s'assurer de la représentativité du groupe témoin dans les deux échantillons en introduisant des coefficients de pondération afin que la proportion pondérée de témoins dans les deux groupes devienne la même.

# %%
prop_gross = (331 + 564) / (90 + 92)
prop_pet = (11 + 94)  / (147 + 671)
coef = prop_gross / prop_pet
coef

# %%
# Drug treatment % of recovery
761/(239+761)

# %%
# Surgery treatment, weighted % of recovery
(564 + 94*coef) / ((331 + 11*coef) + (564 + 94*coef))


