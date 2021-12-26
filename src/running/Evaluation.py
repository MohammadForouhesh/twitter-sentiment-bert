from src.metrics.Metrics import categorical_acc
import torch


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for seq, label in iterator:
            preds = model(seq)
            loss = criterion(preds, label)
            acc = categorical_acc(preds, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)