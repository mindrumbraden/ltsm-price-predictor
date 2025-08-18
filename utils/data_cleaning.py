import pandas as pd

def clean_df(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Reduce dataframe to only two columns:
      1. 'date' column (if datetime exists, else numeric index)
      2. target column (to be predicted)

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Column to keep for prediction

    Returns:
        pd.DataFrame: Cleaned dataframe with ['date', target_col]
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe. Available: {list(df.columns)}")

    # Try to find a datetime column
    datetime_col = None
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            datetime_col = col
            break
        except (ValueError, TypeError):
            continue

    # If datetime column found, use it; else create numeric index
    if datetime_col:
        df['date'] = pd.to_datetime(df[datetime_col])
    else:
        df['date'] = range(len(df))

    # Keep only date + target_col
    df = df[['date', target_col]].copy()

    return df
