import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any, Union, Optional, Tuple
from lightgbm.sklearn import LGBMModel
from lightgbm.basic import Booster

def corr_viz(corr_df: pd.DataFrame,
             threshold: float) -> pd.DataFrame:
    """
    Returns the dataframe of the correlated features, containing only pairs that overpass the `threshold` value.
    Excludes the self-correlation from the results.
    
    Args:
    ----
    corr_df: pd.DataFrame
        Correlation dataframe
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    filtered_df = (corr_df[(corr_df > threshold)]
                   .dropna(how='all', axis=0)
                   .dropna(how='all', axis=1)
                   .stack()
                   .to_frame(name='Pearson corr coef')
                   .sort_values(by='Pearson corr coef', ascending=False)
                   .reset_index())

    return filtered_df[~(filtered_df.level_0 == filtered_df.level_1)].set_index(['level_0', 'level_1']).drop_duplicates().rename_axis([None, None], axis=0)


def custom_feature_imp_plot(booster: Union[Booster, LGBMModel],
                            default_cols: pd.core.indexes.base.Index,
                            encoded_cols: pd.core.indexes.base.Index,
                            importance_type: str = 'auto',
                            figsize: Optional[Tuple[float, float]] = None,) -> Any:
    """
    Summarizes the feature importances of the encoded features.
    E.g. the `provider` feature has several categories, so the Helmert encoder will create a new column for each category. To measure the importance of the original feature, this function sums up the created features.
    
    Args:
    ----
    booster : Booster or LGBMModel
        Booster or LGBMModel instance which feature importance should be plotted.
    default_cols: pd.core.indexes.base.Index
        Columns of the original dataframe (before encoding)
    encoded_cols: pd.core.indexes.base.Index
        Columns of the dataframe after encoding
    importance_type : str, optional (default="auto")
        How the importance is calculated.
        If "auto", if ``booster`` parameter is LGBMModel, ``booster.importance_type`` attribute is used; "split" otherwise.
        If "split", result contains numbers of times the feature is used in a model.
        If "gain", result contains total gains of splits which use the feature.
    figsize : tuple of 2 elements or None, optional (default=None)
        Figure size.
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    
    if isinstance(booster, LGBMModel):
        if importance_type == "auto":
            importance_type = booster.importance_type
        booster = booster.booster_
    elif isinstance(booster, Booster):
        if importance_type == "auto":
            importance_type = "split"
    else:
        raise TypeError('booster must be Booster or LGBMModel.')
    
    df_feat_imp = pd.DataFrame(booster.feature_importance(importance_type=importance_type).reshape(1, -1), columns=encoded_cols)

    df_new_imp = pd.DataFrame()
    list_cols = list(default_cols)

    for col_start_name in list_cols:
        df_new_imp[col_start_name] = df_feat_imp[[col for col in df_feat_imp.columns if col.startswith(col_start_name)]].sum(axis=1)

    df_new_imp = df_new_imp.loc[:, (df_new_imp != 0).any(axis=0)]
    df_new_imp = df_new_imp[df_new_imp.sum().sort_values(ascending=False).index]

    plt.figure(figsize=figsize)
    plt.grid()
    ax = sns.barplot(x=df_new_imp.values.astype(int).ravel(),
                y=df_new_imp.columns,)
    ax.bar_label(ax.containers[0])
    
    return ax