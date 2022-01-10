from src.metrics.Metrics import categorical_acc
from src.params import device
import torch


def train(model, iterator, optimizer, criterion, if_lstm=False):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for seq, label in iterator:
        optimizer.zero_grad()
        if if_lstm: model.hidden_cell = (torch.rand(model.lstm.num_layers, seq.shape[0], model.hidden_layer_size).to(device),
                                         torch.rand(model.lstm.num_layers, seq.shape[0], model.hidden_layer_size).to(device))
        preds = model(seq)
        loss = criterion(preds, label)
        acc = categorical_acc(preds, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)