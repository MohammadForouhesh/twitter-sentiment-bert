from ignite.metrics import Precision, Recall, Accuracy
import torch
import time


def ir_metrics(model, iterator):
    precision = Precision()
    recall = Recall()
    acc = Accuracy()
    
    # Start accumulation:
    for seq, label in iterator:
        y_pred = model(seq)
        precision.update((y_pred, label))
        recall.update((y_pred, label))
        acc.update((y_pred, label))
    
    print("Precision: ", precision.compute(), '\n', precision.compute().mean(), '\n')
    print("Recall: ", recall.compute(), '\n', recall.compute().mean(), '\n')
    print("Accuracy : ", acc.compute(), '\n')


def categorical_acc(preds, label):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    return correct.sum() / torch.FloatTensor([label.shape[0]])


def time_per_epoch(st, et):
    elt = et - st
    elasp_min = int(elt/60)
    elasp_sec = int(elt - elasp_min*60)
    return elasp_min, elasp_sec