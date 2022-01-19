from ignite.metrics import Precision, Recall, Accuracy
import torch
import time


def ir_metrics(model, iterator):
    precision = Precision()
    recall = Recall()
    acc = Accuracy()
    
    # Start accumulation:
    for dl in iterator:
        input_ids      = dl['input_ids']
        attention_mask = dl['attention_mask']
        token_type_ids = dl['token_type_ids']
        targets        = dl['targets']

        y_pred = model(input_ids, attention_mask, token_type_ids)
        precision.update((y_pred, targets))
        recall.update((y_pred, targets))
        acc.update((y_pred, targets))
    
    print("Precision: ", precision.compute(), '\n', precision.compute().mean(), '\n')
    print("Recall: ", recall.compute(), '\n', recall.compute().mean(), '\n')
    print("Accuracy : ", acc.compute(), '\n')


def categorical_acc(preds, label):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    return correct.sum() / torch.cuda.FloatTensor([label.shape[0]])


def time_per_epoch(st, et):
    elt = et - st
    elasp_min = int(elt/60)
    elasp_sec = int(elt - elasp_min*60)
    return elasp_min, elasp_sec