import torch.nn.functional as F
from torch import nn
import torch.nn
import torch

from src.model import LSTM, CNN
from src.model.Cnn import CNN2d


class GCNN(nn.Module):
    def __init__(self, input_size, output_size, n_filters=100, filter_sizes=(1, 2, 2),
                 dropout=0.5, hidden_layer_size=100, bidirectional=False, n_layers=1):
        super().__init__()
        self.lstm = LSTM(input_size, output_size, hidden_layer_size=hidden_layer_size,
                         bidirectional=bidirectional, n_layers=n_layers, dropout=dropout)
        self.cnn1 = CNN(input_size, output_size, n_filters=n_filters)#, filter_sizes=filter_sizes, dropout=dropout)
        self.cnn2 = CNN2d(input_size, output_size, n_filters=n_filters, filter_sizes=filter_sizes, dropout=dropout)
        self.fc = nn.Linear(output_size * 3, output_size)

    def forward(self, text):
        cnn1_out = self.cnn1(text)
        cnn2_out = self.cnn2(text)
        lstm_out = self.lstm(text)
        prediction = self.fc(torch.cat([lstm_out, cnn2_out, cnn1_out], dim=1))
        return prediction