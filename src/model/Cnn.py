import torch.nn.functional as F
import torch
import torch.nn
from torch import nn

trick_f = lambda tensor: tensor.permute(1, 0)\
                               .unsqueeze(-1)\
                               .expand(384, 1, 4)\
                               .unsqueeze(-1)\
                               .expand(384, 1, 4, 100)


class CNN(nn.Module):
    def __init__(self, embedding_dim=100, n_filters=384,
                 filter_sizes=[2, 3, 4], output_dim=42, drop_out=0.5, pad_idx=2):
        super().__init__()
        self.convs = nn.ModuleList([
                     nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(fs, embedding_dim))
                     for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.linear = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, text):
        embedded = trick_f(text)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        out_matrix = self.fc(cat)
        return self.linear(out_matrix).permute(1, 0)