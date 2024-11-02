import joblib
import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from itertools import chain

utils_dir = '/kaggle/input/store-sales-time-series-forecasting-utils'
dataset_dir = '/kaggle/input/store-sales-time-series-forecasting'

from conda.cli import main

main('conda', 'install', '-y',  'pandas=1.3.5', '-c', 'conda-forge')

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split

sys.path.append(utils_dir)

from utils import (
    add_day_of_week,
    add_day_of_month,
    add_day_of_year,
    convert_dummies,
    load_data_frame,
)

with open(os.path.join(utils_dir, 'config.yml')) as f:
    config = yaml.safe_load(f)

X_cols = config['x_cols']
dummy_cols = config['dummy_cols']
random_forest_parameters = config['random_forest_parameters']


def get_data():
    print(f'{datetime.now()} loading training dataframe')
    df = load_data_frame(dataset_dir)
    df = add_day_of_week(df)
    df = add_day_of_month(df)
    df = add_day_of_year(df)

    print(f'{datetime.now()} converting dummies')
    df = convert_dummies(df, dummy_cols)

    print(f'{datetime.now()} train val split')
    train_X, val_X, train_y, val_y = train_test_split(
        df[X_cols], df['sales'], test_size=0.2, random_state=0
    )

    return train_X, val_X, train_y, val_y


def test_training_function():
    train_X, val_X, train_y, val_y = get_data()
    start = datetime.now()

    model = RandomForestRegressor(**random_forest_parameters)

    model.fit(train_X, train_y)

    y_pred = model.predict(val_X)
    rmlse = root_mean_squared_log_error(val_y, y_pred)
    print(f'Validation RMLSE: {rmlse}')
    completion_time = datetime.now() - start

    
    with open(os.path.join('var', 'model.pkl'), 'wb') as f:
        joblib.dump(model, f)

    with open(os.path.join('var', 'training_metrics.json'), 'w') as f:
        json.dump({
            'RMLSE': rmlse,
            'completion_time_seconds': completion_time.total_seconds(),
        }, f)

    return model


def get_test_data():
    print(f'{datetime.now()} loading test dataframe')
    df = load_data_frame(dataset_dir, 'test.csv')
    df = add_day_of_week(df)
    df = add_day_of_month(df)
    df = add_day_of_year(df)

    print(f'{datetime.now()} converting dummies')
    df = convert_dummies(df, dummy_cols)

    return df[X_cols]


def test_model(model, df_X):
    preds = model.predict(df_X)
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')
    subparsers.add_parser('train')
    subparsers.add_parser('infer')
    args = parser.parse_args()

    if args.subparser_name == 'train':
        test_training_function()
        print('Training completed!')
    elif args.subparser_name == 'infer':
        with open(os.path.join('var', 'model.pkl'), 'rb') as f:
            model = joblib.load(f)
        df_X = get_test_data()
        preds = pd.Series(model.predict(df_X), name='sales').rename_axis('id')
        preds.to_csv('submission.csv')
