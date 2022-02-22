import torch.nn as nn


class Sales_RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(Sales_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.flatten = nn.Flatten(0, -2)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = self.flatten(r_out)

        r_out = self.fc(r_out)
        output = r_out.reshape(batch_size, -1)

        return output, hidden
