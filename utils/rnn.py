import torch.nn as nn


class Sales_RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, additional_linear_layers):
        super(Sales_RNN, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lin_layers = [hidden_dim] + additional_linear_layers + [1]

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        self.dropout = nn.Dropout(0.2)

        self.flatten = nn.Flatten(0, -2)

        self.fc = nn.Sequential(
            *(
                nn.Linear(self.lin_layers[i], self.lin_layers[i + 1])
                for i in range(len(self.lin_layers) - 1)
            )
        )

    def forward(self, x, hidden):
        batch_size = x.size(0)

        r_out, hidden = self.rnn(x, hidden)
        r_out = self.dropout(r_out)
        r_out = self.flatten(r_out)

        r_out = self.fc(r_out)
        output = r_out.reshape(batch_size, -1)

        return output, hidden
