from sklearn.metrics import f1_score
from src.estimation.custom_cv import custom_cross_val
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from typing import List, Callable, Dict, Any

def baseline(X, y,
             est_list: List[BaseEstimator],
             shuffle: bool=True,
             random_state: int=None,
             scoring: Callable=f1_score) -> Dict[str, Any]:
    """
    Wraps a custom cross_validation for a list of estimators.
    
    Args:
    ----
    est_list: List[BaseEstimator]
        List of estimators to be fitted
    X
        Independent features
    y
        Dependent features
    n_splits : int, default=3
        Number of folds. Must be at least 2
    shuffle : bool, default=True
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
    res_dict: Dict[str, Any]
        Mean score for each estimator
    """
    res_dict = dict()
    for est in est_list:
        res_dict[estimator_name(est)] = custom_cross_val(est,
                                                        X,
                                                        y,
                                                        random_state=random_state,
                                                        shuffle=shuffle,
                                                        scoring=scoring)[0].mean()
    
    return res_dict

def estimator_name(est: BaseEstimator) -> str:
    """
    Get the name of the ML model from the estimator's class instance
    
    Args:
    ----
    est: BaseEstimator
        sklearn estimator
    Returns:
    -------
    class.name:
        name of the ml algorithm
    """
    if isinstance(est, Pipeline):
        # treat svc with different kernels separately
        try:
            return est[1].__class__.__name__ + '-' + est[1].kernel
        except AttributeError:
            return est[1].__class__.__name__
    else:
        return est.__class__.__name__