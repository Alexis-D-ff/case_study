from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from typing import Dict, Any

from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution
    
)

RND_SEED = 42

# Path to the data
QUOTES = r'Data\long_quotes.csv'

# Modeling pipelines
PP_W_SCLNG = [Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='poly', verbose=True))]),
              Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', verbose=True))]),
              Pipeline([('scaler', StandardScaler()), ('log-reg', LogisticRegression())]),
              Pipeline([('scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])]

PP_WO_SCLNG = [RandomForestClassifier(random_state=RND_SEED, verbose=True),
               LGBMClassifier(random_state=RND_SEED, verbose=-1)]

# Optuna hyperparameter space
OPT_HYPER_PARAMS: Dict[str, Dict[str, Any]] = {
        "learning_rate": FloatDistribution(1e-4, 1e1, log=True),
        "max_depth": IntDistribution(1, 8),
        "n_estimators": IntDistribution(8, 256),
}