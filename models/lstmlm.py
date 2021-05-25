from opacus.layers import DPLSTM
import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.n_vocab, params.embedding_dim)
        assert not params.private
        if params.private or params.log_gradients:
            self.lstm = DPLSTM(input_size=params.embedding_dim, hidden_size=params.hidden_dim, num_layers=params.num_layers)
        else:
            self.lstm = nn.LSTM(input_size=params.embedding_dim, hidden_size=params.hidden_dim, num_layers=params.num_layers)
        self.prediction = nn.Linear(params.embedding_dim, params.n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)

        return self.prediction(output)