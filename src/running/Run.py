from src.metrics.Metrics import time_per_epoch
from src.running.Evaluation import evaluate
from src.running.Train import train
from src.params import N_EPOCH
import torch
import time
import tqdm

best_validation_loss = float('inf')


def run(model, train_iterator, eval_iterator, optimizer, loss_function, n_epoch):
    global best_validation_loss
    start_time = time.time()
    early_stop = False
    for epoch in tqdm.tqdm(range(n_epoch)):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion=loss_function)
        
        valid_loss, valid_acc = evaluate(model, eval_iterator, criterion=loss_function)

        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), f'models/sentiment_{model.__class__.__name__}.pt')
        if valid_acc > 0.8 and abs(valid_loss - train_loss) < 0.01: early_stop = True
        if (epoch + 1) % 10 != 0 and not early_stop: continue

        end_time = time.time()
        epoch_mins, epoch_secs = time_per_epoch(start_time, end_time)
        print(f'Epoch {epoch + 1}, Time: {epoch_mins} mins: {epoch_secs} secs')
        print(f'\t Train Loss {train_loss:.3f}, Train Acc {train_acc * 100:.3f}')
        print(f'\t Valid Loss {valid_loss:.3f}, Valid Acc {valid_acc * 100:.3f}')
        start_time = time.time()
        if early_stop: break
    return model
