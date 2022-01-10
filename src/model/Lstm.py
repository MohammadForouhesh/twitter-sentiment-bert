from src.params import device
from torch import nn
import torch.nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size=150, output_size=10, bidirectional=False, n_layers=5, dropout=0.8):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True,
                            dropout=0 if n_layers < 2 else dropout)

        self.hidden_cell = (torch.rand(1, 1, self.hidden_layer_size).to(device),
                            torch.rand(1, 1, self.hidden_layer_size).to(device))

        self.linear = nn.Linear(hidden_layer_size * 2 if bidirectional else hidden_layer_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vector):
        _, hidden = self.lstm(vector.view(len(vector), 1, -1))
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2::], hidden[-1::]), dim=1))
        else:
            hidden = self.dropout(hidden[0][-1::])
        prediction = self.linear(hidden.view(len(vector), -1))
        return prediction
