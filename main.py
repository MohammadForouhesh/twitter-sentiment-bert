from src.preprocessing.Preprocessing import correction, preprocess, remove_redundent_characters
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from src.model.Lr import LinearRegression
from src.params import device, emb_model
from torch.utils.data import DataLoader
from src.metrics import ir_metrics
from src.model import LSTM, CNN
from src.running.Run import run
from termcolor import colored
from datetime import datetime
from torch import nn
import pandas as pd
import numpy as np
import argparse
import warnings
import logging
import torch
import gc


gc.enable()
warnings.filterwarnings("ignore", category=SyntaxWarning)


def preparation(dataframe, augment=False) -> (np.ndarray, np.ndarray):
    text = dataframe.full_text
    label = dataframe.label.apply(lambda item: item + 1)
    if augment:
        text1 = text.apply(preprocess)
        text2 = pd.concat([text1.apply(lambda sent: ' '.join(sent.split(' ')[::-1])), text], ignore_index=True)
        raw_text = text.apply(remove_redundent_characters)

        text3 = pd.concat([raw_text, text2], ignore_index=True)

        text = pd.concat([raw_text.apply(lambda sent: ' '.join(sent.split(' ')[::-1])), text3], ignore_index=True)
        label = pd.concat([label, label, label, label], ignore_index=True)

    train = list(text.apply(emb_model.encode))

    return train, label


def main(args):
    df = pd.read_csv(args.train_path)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=list(df.label))
    X_train, y_train = preparation(train_df, augment=False)#args.augment)
    X_test, y_test = preparation(test_df, augment=False)

    inputs = torch.from_numpy(np.array(X_train)).to(device)
    if device == 'cuda':    target = torch.cuda.LongTensor(y_train)
    else:                   target = torch.LongTensor(y_train)

    train_ds = TensorDataset(inputs, target)

    inputs = torch.from_numpy(np.array(X_test)).to(device)
    if device == 'cuda':    target = torch.cuda.LongTensor(y_test)
    else:                   target = torch.LongTensor(y_test)
    test_ds = TensorDataset(inputs, target)

    bach_size = 5 if args.model_name == 'lstm' else 1

    train_dl = DataLoader(train_ds, bach_size, shuffle=True)
    test_dl = DataLoader(test_ds, bach_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n====================TRAIN=' + args.model_name.upper() + '=====================', 'red'))
    if args.model_name == 'lstm':
        model = LSTM(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    
        trained_model = run(model=model, iterator=train_dl, optimizer=optimizer,
                            loss_function=loss_function, n_epoch=args.epoch, if_lstm=True)

    elif args.model_name == 'cnn':
        model = CNN(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    
        trained_model = run(model=model, iterator=train_dl, optimizer=optimizer,
                            loss_function=loss_function, n_epoch=args.epoch)

    elif args.model_name == 'lr':
        model = LinearRegression(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)

        trained_model = run(model=model, iterator=train_dl, optimizer=optimizer,
                            loss_function=loss_function, n_epoch=args.epoch)

    ir_metrics(model=trained_model, iterator=train_dl)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n====================Test==' + args.model_name + '=====================', 'red'))

    ir_metrics(model=trained_model, iterator=test_dl)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Architectures')
    parser.add_argument('--train_path', dest='train_path', type=str, default='dataset/bime_sample_new3.csv',
                        help='Raw dataset file address.')
    parser.add_argument('--augment', dest='augment', type=bool, default=True,
                        help='augment the dataset to learn better.')
    parser.add_argument('--model_name', dest='model_name', type=str, default='lstm',
                        help="supported models in this implementation are CNN and LSTM.")
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing the training set.")
    parser.add_argument('--epoch', dest='epoch', type=int, default=30,
                        help="number of epochs in the training")
    parser.add_argument('--test_path', dest='test_path', type=str, default='dataset/EmotionTest.csv',
                        help="address to test dataset.")

    args = parser.parse_args()

    with warnings.catch_warnings():
        logging.basicConfig(filename='neural-arch.log', format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        warnings.filterwarnings("ignore")
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
              colored('\n===============NEURAL=ARCH===============', 'red'))
        main(args)
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'))
