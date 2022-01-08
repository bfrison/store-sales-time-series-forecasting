import torch.nn as nn

class Sales_RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(Sales_RNN, self).__init__
