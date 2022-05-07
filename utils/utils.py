from os.path import join

import numpy as np
import pandas as pd


def load_data_frame(data_path, file_name='train.csv'):
    df_oil = pd.read_csv(
        join(data_path, 'oil.csv'), index_col=0, parse_dates=['date']
    ).convert_dtypes()

    date_min = df_oil.index.min()
    date_max = df_oil.index.max()

    df_oil = df_oil.reindex(pd.date_range(date_min, date_max))
    df_oil.dcoilwtico.fillna(method='ffill', inplace=True)
    df_oil.dcoilwtico.fillna(method='bfill', inplace=True)

    df_holiday_events = pd.read_csv(
        join(data_path, 'holidays_events.csv'), parse_dates=['date']
    ).convert_dtypes()

    df_national_holidays = (
        pd.get_dummies(
            df_holiday_events.query(
                'locale == "National" and not transferred and type != "Work Day"',
                engine='python'
            )[['date', 'type']],
            prefix='National',
            columns=['type'],
        )
        .drop_duplicates()
        .groupby('date')
        .sum()
    ).convert_dtypes()

    df_regional_holidays = (
        df_holiday_events.query('locale == "Regional"', engine='python')
        .groupby(['date', 'locale_name'])
        .description.count()
        .rename('Regional_Holiday')
        .astype('UInt8')
        .to_frame()
    )

    df_local_holidays = (
        pd.get_dummies(
            df_holiday_events.query('locale == "Local" and not transferred', engine='python')[
                ['date', 'locale_name', 'type']
            ],
            prefix='Local',
            columns=['type'],
        )
        .groupby(['date', 'locale_name'])
        .sum()
    ).convert_dtypes()

    df_stores = pd.read_csv(
        join(data_path, 'stores.csv'),
        index_col=0,
        dtype={
            'city': 'category',
            'state': 'category',
            'type': 'category',
            'cluster': 'category',
        },
    ).convert_dtypes()

    # df_transactions = pd.read_csv(join(data_path, 'transactions.csv'), parse_dates=['date'])

    df_train = pd.read_csv(
        join(data_path, file_name),
        index_col=0,
        parse_dates=['date'],
        dtype={'family': 'category'},
    ).convert_dtypes()

    df_train = df_train.join(df_oil, on='date')
    df_train = df_train.join(df_stores, on='store_nbr')
    df_train.store_nbr = df_train.store_nbr.astype('category')

    df_train = df_train.join(df_national_holidays, on='date')
    df_train[df_national_holidays.columns] = df_train[
        df_national_holidays.columns
    ].fillna(0)

    df_train = df_train.join(df_regional_holidays, on=['date', 'state'])
    df_train[df_regional_holidays.columns] = df_train[
        df_regional_holidays.columns
    ].fillna(0)
    df_train.state = df_train.state.astype('category')

    df_train = df_train.join(df_local_holidays, on=['date', 'city'])
    df_train[df_local_holidays.columns] = df_train[df_local_holidays.columns].fillna(0)
    df_train.city = df_train.city.astype('category')

    return df_train


def sequences_generator(df, sequence_length, X_cols, y_col=None):
    num_dates = df.date.nunique()
    sequences_X = []
    sequences_y = []
    for family in df.family.unique():
        df_family = df.query(f'family == @family', engine='python')
        for i in df_family.store_nbr.unique():
            df_store = df_family.query('store_nbr == @i', engine='python')
            for j in range(int(num_dates / sequence_length)):
                index_range = slice(j * sequence_length, (j + 1) * sequence_length)
                sequences_X.append(df_store[X_cols].iloc[index_range].astype('float'))
                if y_col:
                    sequences_y.append(
                        df_store[y_col].iloc[index_range].astype('float')
                    )
                else:
                    sequences_y.append(df_store.iloc[index_range].index.astype('int'))

    return np.array(sequences_X), np.array(sequences_y)


def convert_dummies(df, cols, reattach_family=True):
    df_dummies_list = []
    for col in cols:
        df_dummies_list.append(pd.get_dummies(df[col], prefix=col).convert_dtypes())

    if reattach_family:
        family = df.family

        return pd.concat([df.drop(columns=cols)] + [family] + df_dummies_list, axis=1)
    else:

        return pd.concat([df.drop(columns=cols)] + df_dummies_list, axis=1)


def add_day_of_week(df, col_name='date'):
    df['weekday'] = df[col_name].dt.day_name()

    return df


def add_quarter(df, col_name='date'):
    df['quarter'] = df[col_name].dt.quarter.astype('category')
    df['quarter'].cat.set_categories([1, 2, 3, 4], inplace=True)

    return df
