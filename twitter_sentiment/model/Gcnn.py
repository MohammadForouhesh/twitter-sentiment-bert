from torch import nn
import torch.nn
import torch

from twitter_sentiment.model import LSTM, CNN
from twitter_sentiment.model.Cnn import CNN2d


class GCNN(nn.Module):
    def __init__(self, input_size, output_size, n_filters=50, filter_sizes=(1, 2, 2),
                 dropout=0.5, hidden_layer_size=100, bidirectional=False, n_layers=1):
        super().__init__()
        self.lstm = LSTM(input_size, output_size, hidden_layer_size=hidden_layer_size,
                         bidirectional=bidirectional, n_layers=n_layers, dropout=dropout)
        self.cnn1 = CNN(hidden_layer_size*2, output_size, n_filters=n_filters)#, filter_sizes=filter_sizes, dropout=dropout)
        self.cnn2 = CNN2d(hidden_layer_size*2, output_size, n_filters=n_filters, filter_sizes=filter_sizes, dropout=dropout)
        self.fc = nn.Linear(output_size * 2, output_size)
        self.dropout = nn.FeatureAlphaDropout(p=0.2)

    def forward(self, text):
        _ = self.lstm(text)
        memory = self.dropout(torch.cat(self.lstm.hidden_cell, dim=2).squeeze(0))

        cnn1_out = self.cnn1(memory)
        cnn2_out = self.cnn2(memory)
        prediction = self.fc(torch.cat([cnn2_out, cnn1_out], dim=1))
        return prediction