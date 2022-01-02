from os.path import join

import pandas as pd


def load_data_frame(data_path):
    df_holiday_events = pd.read_csv(
        join(data_path, 'holidays_events.csv'), parse_dates=['date']
    ).convert_dtypes()

    df_oil = pd.read_csv(join(data_path, 'oil.csv'), index_col=0, parse_dates=['date']).convert_dtypes()
    df_oil = df_oil.reindex(pd.date_range(df_oil.index.min(), df_oil.index.max()))
    df_oil.dcoilwtico.fillna(method='ffill', inplace=True)
    df_oil.dcoilwtico.fillna(method='bfill', inplace=True)

    df_stores = pd.read_csv(join(data_path, 'stores.csv'), index_col=0, dtype={'city': 'category', 'state': 'category', 'type': 'category', 'cluster': 'category'}).convert_dtypes()

    # df_transactions = pd.read_csv(join(data_path, 'transactions.csv'), parse_dates=['date'])

    df_train = pd.read_csv(
        join(data_path, 'train.csv'), index_col=0, parse_dates=['date'], dtype={'family': 'category'}
    ).convert_dtypes()

    df_train = df_train.join(df_oil, on='date')
    df_train = df_train.join(df_stores, on='store_nbr')

    return df_train
