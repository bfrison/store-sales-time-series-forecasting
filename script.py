from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from rnn import Sales_RNN
from utils import load_data_frame, sequences_generator


def get_data():
    df = load_data_frame('var')

    X_cols = ['National_Holiday', 'National_Event']

    sequences_X, sequences_y = sequences_generator(df, 'AUTOMOTIVE', 10, X_cols)

    train_X, val_X, train_y, val_y = train_test_split(
        sequences_X, sequences_y, test_size=0.2, random_state=0
    )

    return train_X, val_X, train_y, val_y


def train(model, train_dataloader, epochs, criterion, optimizer, print_every=100):
    start = datetime.now()

    for batch_X, batch_y in train_dataloader:

        hidden = None
        optimizer.zero_grad()
        batch_y_log = np.log(batch_y + 1)
        prediction, hidden = model(batch_X, hidden)
        loss = criterion(prediction, batch_y_log)
        loss.backward()
        optimizer.step()

        return model


def test_training_function():
    train_X, val_X, train_y, val_y = get_data()
    train_dataset = TensorDataset(
        torch.from_numpy(train_X[:128].astype('float32')),
        torch.from_numpy(train_y[:128].astype('float32')),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    model = Sales_RNN(2, 8, 3)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    model = train(model, train_loader, 2, criterion, optimizer)
