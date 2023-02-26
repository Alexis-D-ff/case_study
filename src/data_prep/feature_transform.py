import pandas as pd
import numpy as np
from typing import List

def set_dtypes(df: pd.DataFrame,
               cat_cols: List[str],
               bool_cols: List[str],
               datetime_cols: List[str]) -> pd.DataFrame:
    """
    Transforms the initial datatype of the provided columns according to their nature. 
    
    Args:
    ----
    df: pd.DataFrame
        Initial dataframe
    cat_cols: List[str]
        List of the categorical columns
    bool_cols: List[str]
        List of the boolean columns
    datetime_cols: List[str]
        List of the datetime columns
    
    Returns:
    -------
    res_df: pd.DataFrame
        The dataframe with re-initialized dtypes
    """
    res_df = df.copy(deep=True)
    res_df[cat_cols] = res_df[cat_cols].astype(object)
    res_df[bool_cols] = res_df[bool_cols].apply(pd.to_numeric)
    res_df[datetime_cols] = res_df[datetime_cols].apply(pd.to_datetime, utc=True)
    
    return res_df

def slice_nan_imputer(df: pd.DataFrame,
                      list_cols: List[str],
                      string: str) -> pd.DataFrame:
    """
    Impute NaN values with the passed `string` keyword on the slice of a dataframe.
    
    Args:
    ----
    df: pd.DataFrame
        Initial dataframe
    list_cols: List[str]
        List of the columns to be treated
    string: str
        Keyword to be used to replace NaN values
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    res_df = df.copy(deep=True)
    res_df[list_cols] = res_df[list_cols].fillna(string, downcast=False)
    
    return res_df

def col_binarize(df: pd.DataFrame,
                 list_cols: List[str]) -> pd.DataFrame:
    """
    Binarizes the column values for a list of columns. All NaNs are re-defined as 0s, all other values will be replaced with 1s (no matter which value was initially).
    This is useful for columns with high amount of NaNs. 
    
    Args:
    ----
    df: pd.DataFrame
        Initial dataframe
    list_cols: List[str]
        List of the columns to be treated
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    res_df = df.copy(deep=True)
    res_df[list_cols] = (res_df[list_cols].notna()).astype(int)
    
    return res_df


def esd_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the NaNs of the `effective_start_date` column.
    Add the median datediff of the `effective_start_date` and the `submitted_at` features, computed for the full dataset, to the value of the `submitted_at` column (no NaNs should be present).
    
    Args:
    ----
    df: pd.DataFrame
        Initial dataframe
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    res_df = df.copy(deep=True)
    imp_val = (res_df['effective_start_date'] - res_df['submitted_at']).clip(lower='0 days').median()
    res_df['effective_start_date'] = res_df['effective_start_date'].transform(lambda row: row.fillna(res_df['submitted_at'] + imp_val))
    
    return res_df

def datetime_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the datetime columns of the provided dataset.
    New features will be created:
        col_year - int value of the parsed year
        col_month - cat month name of the parsed month
        col_day - int value of the parsed day of the month
        
        if the time is present for any parsed row:
            col_time_sin - sinus of the time in seconds
            col_time_cos - cosine of the time in seconds
    
    Args:
    ----
    df: pd.DataFrame
        Initial dataframe
    list_cols: List[str]
        List of the columns to be treated
    
    Returns:
    -------
    res_df: pd.DataFrame
        The resulting dataframe
    """
    res_df = df.copy(deep=True)
    seconds_in_day = 24*60*60
    for col in res_df.select_dtypes(include='datetime64[ns, UTC]').columns:
        res_df[col + '_year'] = res_df[col].dt.year
        res_df[col + '_month'] = res_df[col].dt.month_name()
        res_df[col + '_day'] = res_df[col].dt.day
        total_seconds = (res_df[col].dt.hour*60+res_df[col].dt.minute)*60 + res_df[col].dt.second
        if not (total_seconds == 0).all():
            res_df[col + '_time_sin'] = np.sin(2*np.pi*total_seconds/seconds_in_day)
            res_df[col + '_time_cos'] = np.cos(2*np.pi*total_seconds/seconds_in_day)
        res_df.drop(col, axis=1, inplace=True)
    
    return res_df