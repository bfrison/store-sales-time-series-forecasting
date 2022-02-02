from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from rnn import Sales_RNN
from utils import convert_dummies, load_data_frame, sequences_generator

with open('config.yml') as f:
    config = yaml.safe_load(f)

X_cols = config['x_cols']
dummy_cols = config['dummy_cols']


def get_data():
    df = load_data_frame('var')
    df = convert_dummies(df, dummy_cols)

    sequences_X, sequences_y = sequences_generator(df, 'AUTOMOTIVE', 10, X_cols)

    train_X, val_X, train_y, val_y = train_test_split(
        sequences_X, sequences_y, test_size=0.2, random_state=0
    )

    return train_X, val_X, train_y, val_y


def train(
    model,
    train_dataloader,
    epochs,
    criterion,
    optimizer,
    print_every=100,
    device='cuda',
):
    start = datetime.now()
    print(f'Training started at {start}')

    model.train()

    for epoch in range(epochs):

        for index, (batch_X, batch_y) in enumerate(train_dataloader):
            hidden = None
            optimizer.zero_grad()

            batch_y_log = np.log(batch_y + 1)
            batch_X, batch_y_log = batch_X.to(device), batch_y_log.to(device)

            prediction, hidden = model(batch_X, hidden)
            loss = criterion(prediction, batch_y_log)
            loss.backward()
            optimizer.step()

            if (index + 1) % print_every == 0:
                print(
                    f'\t{index + 1} batches completed, time elapsed: {datetime.now() - start}'
                )

        print(
            f'Epoch {epoch+1:2d}/{epochs:d} completion time: {datetime.now() - start}'
        )

    print(f'Training completion time: {datetime.now() - start}')

    return model


def test_training_function():
    train_X, val_X, train_y, val_y = get_data()
    train_dataset = TensorDataset(
        torch.from_numpy(train_X.astype('float32')),
        torch.from_numpy(train_y.astype('float32')),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    model = Sales_RNN(len(X_cols), 8, 3)
    device = 'cuda'
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    model = train(model, train_loader, 2, criterion, optimizer, device=device)

    return model
