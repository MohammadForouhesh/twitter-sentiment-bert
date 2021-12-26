from src.params import device
from torch import nn
import torch.nn


class LSTM(nn.Module):
    def __init__(self, input_size=384, hidden_layer_size=150, output_size=10):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=1)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        self.hidden_cell = (torch.rand(1, 1, self.hidden_layer_size).to(device),
                            torch.rand(1, 1, self.hidden_layer_size).to(device))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        return predictions