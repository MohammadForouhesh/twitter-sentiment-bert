from twitter_sentiment.metrics.Metrics import categorical_acc
import torch


def train(model, iterator, optimizer, criterion, scheduler, clip=0.0):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for dl in iterator:
        input_ids      = dl['input_ids']
        attention_mask = dl['attention_mask']
        token_type_ids = dl['token_type_ids']
        targets        = dl['targets']

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(preds, targets)
        acc = categorical_acc(preds, targets)
        loss.backward()

        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)