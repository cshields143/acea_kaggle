import pandas as pd
import miceforest as mf
from datetime import datetime

def drop_where_null(df, col):
    return df[~df[col].isnull()]

def date_as_index(df, col, format):
    df.index = pd.to_datetime(df[col], format=format)
    df.index.name = None
    df.index.freq = 'D'
    return df.drop(col, axis=1)

def start_after(df, y, m, d):
    return df[df.index > datetime(y, m, d)]

# engineering features
# ...

def trim_to_targets(df, y):
    notnulls = [df[~df[c].isnull()] for c in y]
    ends = [(nn.iloc[0].name, nn.iloc[-1].name) for nn in notnulls]
    firsts, lasts = zip(*ends)
    return df.loc[max(firsts):min(lasts)]

def reimpute_targets(df, y):
    for c in y:
        df[c] = df[c].replace(0, float('nan'))
    return df

def discard_sparse_features(df, x, thresh):
    todrop = []
    for c in x:
        ratio = df[c].isnull().sum() / df.shape[0]
        if ratio >= thresh:
            todrop.append(c)
    return df.drop(todrop, axis=1)

def absolute_columns(df, begs):
    cols = sum([[c for c in df.columns if c.startswith(b)] for b in begs], [])
    for c in cols:
        df[c] = df[c].abs()
    return df

def multiply_impute(df):
    kernel = mf.MultipleImputedKernel(
        data=df,
        save_all_iterations=False,
        random_state=143
    )
    kernel.mice(3, verbose=False)
    return kernel.impute_new_data(df).complete_data(0)

def aquifer_pipe(df, x, y):
    df = date_as_index(df, 'Date', '%d/%m/%Y')
    df = trim_to_targets(df, y)
    df = reimpute_targets(df, y)
    df = discard_sparse_features(df, x, .7)
    df = absolute_columns(df, ['Rainfall', 'Volume', 'Depth_to_Groundwater'])
    df = multiply_impute(df)
    return df

def waterspring_pipe(df, x, y):
    df = drop_where_null(df, 'Date')
    df = date_as_index(df, 'Date', '%d/%m/%Y')
    df = trim_to_targets(df, y)
    df = reimpute_targets(df, y)
    df = absolute_columns(df, ['Rainfall', 'Depth_to_Groundwater', 'Flow_Rate'])
    df = multiply_impute(df)
    return df

def river_pipe(df, x, y):
    df = drop_where_null(df, 'Date')
    df = date_as_index(df, 'Date', '%d/%m/%Y')
    df = trim_to_targets(df, y)
    df = reimpute_targets(df, y)
    df = discard_sparse_features(df, x, .7)
    df = absolute_columns(df, ['Rainfall'])
    df = multiply_impute(df)
    return df

def lake_pipe(df, x, y):
    df = date_as_index(df, 'Date', '%d/%m/%Y')
    df = start_after(df, 2004, 1, 1)
    df = absolute_columns(df, ['Rainfall', 'Flow_Rate', 'Lake_Level'])
    return df
