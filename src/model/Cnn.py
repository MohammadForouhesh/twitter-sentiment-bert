import torch.nn.functional as F
from torch import nn
import torch.nn
import torch


class CNN(nn.Module):
    def __init__(self, input_size, output_size, n_filters=100, filter_sizes=(1, 1, 1),
                 dropout=0.4):
        super().__init__()
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels=input_size,
                                              out_channels=n_filters,
                                              kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = text.unsqueeze(2)
        conved = [F.leaky_relu(conv(embedded)) for conv in self.convs]
        pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        predictions = self.fc(cat.view(len(text), -1))
        return predictions


class CNN2d(nn.Module):
    def __init__(self, input_size, output_size, n_filters=20, filter_sizes=(1, 2, 2),
                 dropout=0.5):
        super().__init__()
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels=input_size,
                                              out_channels=n_filters,
                                              kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = text.unsqueeze(2).unsqueeze(3).expand(-1, -1, 32, 32)
        conved = [F.leaky_relu(conv(embedded)) for conv in self.convs2]
        pooled2 = [F.avg_pool2d(conv, conv.shape[3]).squeeze(3) for conv in conved]

        cat = self.dropout(torch.cat(pooled2, dim=2))
        # cat = [batch size, n_filters * len(filter_sizes)]
        predictions = self.fc(cat.view(len(text), -1))
        return predictions


class CNN2d1(nn.Module):
    def __init__(self, input_size, output_size, n_filters=300, filter_sizes=(1, 2, 2),
                 dropout=0.5):
        super().__init__()
        self.convs2 = nn.ModuleList([
                                    nn.Conv2d(in_channels=input_size,
                                              out_channels=n_filters*2,
                                              kernel_size=fs)
                                    for fs in filter_sizes
                                    ])

        self.convs1 = nn.ModuleList([
                                    nn.Conv1d(in_channels=n_filters*2,
                                              out_channels=n_filters,
                                              kernel_size=fs)
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = text.unsqueeze(2).unsqueeze(3).expand(-1, -1, 32, 32)
        conved = [F.leaky_relu(conv(embedded)) for conv in self.convs2]
        pooled2 = [F.avg_pool2d(conv, conv.shape[3]).squeeze(3) for conv in conved]

        cat = self.dropout(torch.cat(pooled2, dim=2))
        conved = [F.leaky_relu(conv(cat)) for conv in self.convs1]
        pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        predictions = self.fc(cat.view(len(text), -1))
        return predictions