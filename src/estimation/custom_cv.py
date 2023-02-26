import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from category_encoders.helmert import HelmertEncoder
from typing import Callable
from sklearn.metrics import recall_score
from typing import Tuple, Any

def custom_cross_val(est: BaseEstimator,
                     X, y,
                     n_splits: int=3,
                     random_state: int=None,
                     shuffle: bool=False,
                     scoring: Callable=recall_score) -> Tuple[Any]:
    """
    Custom cross_validation function, used for applying of the encoding in each fold
    
    Args:
    ----
    est: BaseEstimator
        estimator to be fitted
    X
        Independent features
    y
        Dependent features
    n_splits : int, default=3
        Number of folds. Must be at least 2
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
    scoring: Callable=f1_score
        Function to be used to compute estimator scores
    
    Returns:
    -------
    Tuple
        (scores of each fold, y_true and y_pred of each fold, encoder instance of each fold)
    """
    cat_features = list(X.select_dtypes(include=[object], exclude=['datetime', 'timedelta']).columns)
    folder = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    true_preds, scores, encoders = list(), list(), list()
    
    for train_ids, test_ids in folder.split(X, y):
        helm_encoder = HelmertEncoder(cols=cat_features)
        train_x = helm_encoder.fit_transform(X.iloc[train_ids, :], y[train_ids])
        est.fit(train_x, y[train_ids])
        
        test_x = helm_encoder.transform(X.iloc[test_ids, :])
        test_y = y[test_ids]
        y_pred = est.predict(test_x)
        true_preds.append([test_y, y_pred])
        scores.append(scoring(test_y, y_pred))
        encoders.append(helm_encoder)
    
    return np.array(scores), true_preds, encoders