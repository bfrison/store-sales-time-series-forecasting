from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from rnn import Sales_RNN
from utils import add_day_of_week, add_quarter, convert_dummies, load_data_frame, sequences_generator

with open('config.yml') as f:
    config = yaml.safe_load(f)

X_cols = config['x_cols']
dummy_cols = config['dummy_cols']
hidden_dimensions = config['network_hyperparameters']['hidden_dimensions']
n_layers = config['network_hyperparameters']['n_layers']
batch_size = config['training_hyperparameters']['batch_size']
num_epochs = config['training_hyperparameters']['num_epochs']
lr = config['training_hyperparameters']['lr']


def get_data():
    print(f'{datetime.now()} loading dataframe')
    df = load_data_frame('var')
    df = add_day_of_week(df)
    df = add_quarter(df)

    print(f'{datetime.now()} converting dummies')
    df = convert_dummies(df, dummy_cols)

    print(f'{datetime.now()} generating sequences')
    sequences_X, sequences_y = sequences_generator(df, 1684, X_cols)

    print(f'{datetime.now()} train test split')
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
    min_mse = np.inf

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
        mse = cumulative_loss / denominator
        print(f'Validation MSE: {mse}')
        if mse < min_mse:
            min_mse = mse
            torch.save(model.state_dict(), 'model.pkl')
            print('Saved model artifacts')

    print(f'Training completion time: {datetime.now() - start}')
    print(f'Lowest MSE: {min_mse}')

    return model


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
    model = Sales_RNN(len(X_cols), hidden_dimensions, n_layers)
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


if __name__ == '__main__':
    test_training_function()
