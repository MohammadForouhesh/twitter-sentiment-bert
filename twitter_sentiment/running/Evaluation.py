from twitter_sentiment.metrics.Metrics import categorical_acc
import torch


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for dl in iterator:
            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']
            targets = dl['targets']

            preds = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(preds, targets)
            acc = categorical_acc(preds, targets)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)