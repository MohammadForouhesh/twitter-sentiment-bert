from src.metrics.Metrics import time_per_epoch
from src.running.Evaluation import evaluate
from src.running.Train import train
from src.params import N_EPOCH
import torch
import time

best_validation_loss = float('inf')


def run(model, iterator, optimizer, loss_function, n_epoch, if_lstm=False):
    global best_validation_loss
    start_time = time.time()

    for epoch in range(n_epoch):
        train_loss, train_acc = train(model, iterator, optimizer, criterion=loss_function, if_lstm=if_lstm)
        
        valid_loss, valid_acc = evaluate(model, iterator, criterion=loss_function)

        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), 'models/exa_emotion_classification.pt')

        if (epoch + 1) % 10 != 0: continue

        end_time = time.time()
        epoch_mins, epoch_secs = time_per_epoch(start_time, end_time)
        print(f'Epoch {epoch + 1}, Time: {epoch_mins} mins: {epoch_secs} secs')
        print(f'\t Train Loss {train_loss:.3f}, Train Acc {train_acc * 100:.3f}')
        print(f'\t Valid Loss {valid_loss:.3f}, Valid Acc {valid_acc * 100:.3f}')
        start_time = time.time()
    return model
