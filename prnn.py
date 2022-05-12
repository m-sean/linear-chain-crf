import torch
from torch import nn


class PRNNModel(nn.Module):
    """Projection-feature based RNN model.

    Args:
        input_dim: Size of the features from the projection encoder.
        fc_dim: Output size of the linear bottleneck layer.
        hidden_dim: Output size of the RNN layer.
        output_dim: Number of classes for output.
        num_layers: Number of RNN layers to use.
        bidirectional: Whether to set the RNN layer to bidirectional.
        dropout: Dropout probability applied after the bottlneck layer.
        rnn_dropout: Dropout probability applied between rnn layers (num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        bidirectional: int,
        dropout: float,
        rnn_dropout: float,
    ):
        super(PRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size=fc_dim,
            hidden_size=hidden_dim,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.fc2 = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        hid = self._init_hidden(x.shape[0])
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        x, _ = self.gru(x, hid)
        x = self.fc2(self.relu(x))
        return x

    def _init_hidden(self, batch_size):
        h0 = self.num_layers * 2 if self.bidirectional else self.num_layers
        hidden = torch.zeros((h0, batch_size, self.hidden_dim))
        return hidden
