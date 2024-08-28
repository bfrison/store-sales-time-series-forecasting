from argparse import ArgumentParser
import json
import os
import sys
from datetime import datetime

utils_dir = '/kaggle/input/store-sales-time-series-forecasting-utils'
dataset_dir = '/kaggle/input/store-sales-time-series-forecasting'

from conda.cli import main

main('conda', 'install', '-y',  'pandas=1.3.5', '-c', 'conda-forge')

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(utils_dir)

from rnn import Sales_RNN
from utils import (
    add_day_of_week,
    add_day_of_month,
    add_day_of_year,
    convert_dummies,
    load_data_frame,
    sequences_generator,
)

with open(os.path.join(utils_dir, 'config.yml')) as f:
    config = yaml.safe_load(f)

X_cols = config['x_cols']
dummy_cols = config['dummy_cols']
hidden_dimensions = config['network_hyperparameters']['hidden_dimensions']
n_layers = config['network_hyperparameters']['n_layers']
additional_linear_layers = config['network_hyperparameters']['additional_linear_layers']
batch_size = config['training_hyperparameters']['batch_size']
num_epochs = config['training_hyperparameters']['num_epochs']
lr = config['training_hyperparameters']['lr']


def get_data():
    print(f'{datetime.now()} loading training dataframe')
    df = load_data_frame(dataset_dir)
    df = add_day_of_week(df)
    df = add_day_of_month(df)
    df = add_day_of_year(df)

    print(f'{datetime.now()} converting dummies')
    df = convert_dummies(df, dummy_cols)

    print(f'{datetime.now()} generating sequences')
    sequences_X, sequences_y = sequences_generator(df, 1684, X_cols, 'sales')

    print(f'{datetime.now()} train val split')
    train_X, val_X, train_y, val_y = train_test_split(
        sequences_X, sequences_y, test_size=0.2, random_state=0
    )

    return train_X, val_X, train_y, val_y


def train(
    model,
    train_dataloader,
    validation_dataloader,
    epochs,
    criterion,
    optimizer,
    print_every=1000,
    device='cuda',
):
    start = datetime.now()
    print(f'Training started at {start}')
    min_rmlse = np.inf

    for epoch in range(epochs):

        model.train()

        for index, (batch_X, batch_y) in enumerate(train_dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            hidden = None
            optimizer.zero_grad()

            batch_y_log = torch.log(batch_y + 1)

            prediction, hidden = model(batch_X, hidden)
            loss = criterion(prediction, batch_y_log)
            loss.backward()
            optimizer.step()

            if (index + 1) % print_every == 0:
                print(
                    f'\t{index + 1} batches completed, time elapsed: {datetime.now() - start}'
                )

        model.eval()

        cumulative_loss = 0
        denominator = 0
        for batch_X, batch_y in validation_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            hidden = None
            denominator += len(batch_y)

            batch_y_log = torch.log(batch_y + 1)

            prediction, hidden = model(batch_X, hidden)
            loss = criterion(prediction, batch_y_log)

            cumulative_loss += loss.item() * len(batch_y)

        print(
            f'Epoch {epoch+1:2d}/{epochs:d} completion time: {datetime.now() - start}',
            end='\t',
        )
        rmlse = cumulative_loss / denominator
        print(f'Validation RMLSE: {rmlse}')
        if rmlse < min_rmlse:
            min_rmlse = rmlse
            if not os.path.exists('var'):
                os.mkdir('var')
            torch.save(model.state_dict(), os.path.join('var', 'model.pkl'))
            print('Saved model artifacts')

    completion_time = datetime.now() - start
    with open(os.path.join('var', 'training_metrics.json'), 'w') as f:
        json.dump({'RMLSE': min_rmlse, 'completion_time_seconds': completion_time.total_seconds()}, f)
    print(f'Training completion time: {completion_time}')
    print(f'Lowest RMLSE: {min_rmlse}')

    return model, completion_time, min_rmlse


def test_training_function():
    train_X, val_X, train_y, val_y = get_data()
    train_dataset = TensorDataset(
        torch.from_numpy(train_X.astype('float32')),
        torch.from_numpy(train_y.astype('float32')),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(val_X.astype('float32')),
        torch.from_numpy(val_y.astype('float32')),
    )
    # print(f'Dataset length: {len(train_dataset):d}')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(
        validation_dataset, shuffle=True, batch_size=batch_size
    )
    model = Sales_RNN(
        len(X_cols), hidden_dimensions, n_layers, additional_linear_layers
    )
    device = 'cuda'
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    model = train(
        model,
        train_loader,
        validation_loader,
        num_epochs,
        criterion,
        optimizer,
        device=device,
    )

    return model


def get_test_data():
    print(f'{datetime.now()} loading test dataframe')
    df = load_data_frame(dataset_dir, 'test.csv')
    df = add_day_of_week(df)
    df = add_day_of_month(df)
    df = add_day_of_year(df)

    print(f'{datetime.now()} converting dummies')
    df = convert_dummies(df, dummy_cols)

    print(f'{datetime.now()} generating sequences')
    sequences_X, sequences_index = sequences_generator(df, 16, X_cols)

    return sequences_X, sequences_index


def test_model(model, sequences_X, sequences_index):
    model.eval()
    sequences_y_log, hidden = model(
        torch.from_numpy(sequences_X.astype('float32')), None
    )
    sequences_y = np.exp(sequences_y_log.detach().numpy()) - 1
    index = sequences_index.reshape(-1)
    preds = (
        pd.Series(sequences_y.reshape(-1))
        .set_axis(index)
        .sort_index()
        .rename('sales')
        .rename_axis('id')
    )
    return preds


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')
    subparsers.add_parser('train')
    subparsers.add_parser('infer')
    args = parser.parse_args()

    if args.subparser_name == 'train':
        test_training_function()
    elif args.subparser_name == 'infer':
        model = Sales_RNN(
            len(X_cols), hidden_dimensions, n_layers, additional_linear_layers
        )
        state_dict = torch.load(os.path.join('var', 'model.pkl'), weights_only=True)
        model.load_state_dict(state_dict)
        sequences_X, sequences_index = get_test_data()
        preds = test_model(model, sequences_X, sequences_index)
        preds.to_csv('submission.csv')
