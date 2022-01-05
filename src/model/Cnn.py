import torch.nn.functional as F
from torch import nn
import torch.nn
import torch


class CNN(nn.Module):
    def __init__(self, input_size, output_size, n_filters=800, filter_sizes=[1, 1, 1, 1, 1],
                 dropout=0.25):
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
        embedded = text.unsqueeze(0)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)