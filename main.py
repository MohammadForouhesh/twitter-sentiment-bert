import pickle5 as pickle

from src.model.Cnn import CNN2d1, CNN2d
from src.model.Gcnn import GCNN
from src.preprocessing.Preprocessing import preprocess, remove_redundent_characters
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
from tqdm import tqdm
from torch import nn
import pandas as pd
import numpy as np
import argparse
import warnings
import logging
import torch
import gc

id2label = {0: 'sad', 1: 'meh', 2: 'happy'}
# label2id = {'sad': 0, 'meh': 1, 'happy': 2}
label2id = {'POSITIVE': 1, 'NEGATIVE': 0}
gc.enable()
warnings.filterwarnings("ignore", category=SyntaxWarning)


def preparation(dataframe, augment=False) -> (np.ndarray, np.ndarray):
    dataframe.fillna('', inplace=True)
    text = dataframe.text#.apply(remove_redundent_characters)
    label = dataframe.sentiment.apply(lambda item: label2id[item])
    if augment:
        text1 = text.apply(preprocess)
        text2 = pd.concat([text1.apply(lambda sent: ' '.join(sent.split(' ')[::-1])), text1], ignore_index=True)
        raw_text = text.apply(remove_redundent_characters)

        text3 = pd.concat([raw_text, text2], ignore_index=True)

        text = pd.concat([raw_text.apply(lambda sent: ' '.join(sent.split(' ')[::-1])), text3], ignore_index=True)
        label = pd.concat([label, label, label, label], ignore_index=True)
    tqdm.pandas()
    train = text.progress_apply(emb_model.encode, batch_size=128, show_progress_bar=False)

    return list(train), list(label)


def inference(model, sentence):
    encoded = emb_model.encode([sentence], convert_to_tensor=True, batch_size=128, show_progress_bar=False)
    return id2label[int(model(encoded)[0].argmax())]


def main(args):
    if args.load is not None:
        trained_model = LSTM(input_size=768, output_size=2).to(device)
        trained_model.load_state_dict(torch.load(args.load))
        trained_model.eval()
        tqdm.pandas()
        inference_set = pd.read_excel(args.test_path)
        inference_set = inference_set.dropna()
        inference_set['sentiment'] = inference_set.text.progress_apply(lambda item: inference(trained_model, item))
        inference_set.to_excel('datasets_polarity_lstm.xlsx')
        return 0
    loading = True
    if not loading:
        df = pd.read_excel(args.train_path).sample(n=2000)
        # negative_data = df[df['sentiment'] == 'sad']
        # neutral_data = df[df['sentiment'] == 'meh']
        # positive_data = df[df['sentiment'] == 'happy']
        #
        # cutting_point = min(len(negative_data), len(neutral_data), len(positive_data))
        # print(cutting_point)
        # if cutting_point <= len(negative_data):
        #     negative_data = negative_data.sample(n=cutting_point).reset_index(drop=True)
        #
        # if cutting_point <= len(neutral_data):
        #     neutral_data = neutral_data.sample(n=cutting_point).reset_index(drop=True)
        #
        # if cutting_point <= len(positive_data):
        #     positive_data = positive_data.sample(n=cutting_point).reset_index(drop=True)

        # df = pd.concat([negative_data, neutral_data, positive_data])
        train_df, test_df = train_test_split(df, test_size=0.15, stratify=list(df.sentiment))
        train_df, eval_df = train_test_split(train_df, test_size=0.1, stratify=list(train_df.sentiment))

        X_train, y_train = preparation(train_df, augment=False)
        X_eval, y_eval = preparation(eval_df, augment=False)
        X_test, y_test = preparation(test_df, augment=False)
        data_pickle = [X_train, y_train, X_eval, y_eval, X_test, y_test]
        with open('sample_normal.pkl', 'wb') as f:
            pickle.dump(data_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        data_pickle = pickle.load(open('unbiased_total.pkl', 'rb'))
        X_train, y_train, X_eval, y_eval, X_test, y_test = data_pickle

    inputs = torch.from_numpy(np.array(X_train)).to(device)
    if device == 'cuda':    target = torch.cuda.LongTensor(y_train)
    else:                   target = torch.LongTensor(y_train)

    train_ds = TensorDataset(inputs, target)

    inputs = torch.from_numpy(np.array(X_eval)).to(device)
    if device == 'cuda':    target = torch.cuda.LongTensor(y_eval)
    else:                   target = torch.LongTensor(y_eval)

    eval_ds = TensorDataset(inputs, target)

    inputs = torch.from_numpy(np.array(X_test)).to(device)
    if device == 'cuda':    target = torch.cuda.LongTensor(y_test)
    else:                   target = torch.LongTensor(y_test)
    test_ds = TensorDataset(inputs, target)

    batch_size = 32

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    eval_dl = DataLoader(eval_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n===============TRAIN=' + args.model_name.upper() + '===============', 'red'))
    if args.model_name == 'lstm':
        model = LSTM(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)

    elif args.model_name == 'cnn':
        model = GCNN(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)

    elif args.model_name == 'lr':
        model = LinearRegression(input_size=len(X_train[0]), output_size=len(set(y_train))).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    trained_model = run(model=model, train_iterator=train_dl, eval_iterator=eval_dl, optimizer=optimizer,
                        loss_function=loss_function, n_epoch=args.epoch)

    ir_metrics(model=trained_model, iterator=train_dl)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n===============Test==' + args.model_name.upper() + '===============', 'red'))

    ir_metrics(model=trained_model, iterator=test_dl)
    """
    inference_set = pd.read_csv(args.test_path)
    inference_set['sentiment'] = inference_set.text.apply(lambda item: inference_model(trained_model, item))
    inference_set.to_csv('datasets_sentiment_lstm.csv')
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Architectures')
    parser.add_argument('--train_path', dest='train_path', type=str, default='dataset/sentiment-RoBerta-finetuning.xlsx',
                        help='Raw dataset file address.')
    parser.add_argument('--augment', dest='augment', type=bool, default=True,
                        help='augment the dataset to learn better.')
    parser.add_argument('--model_name', dest='model_name', type=str, default='cnn',
                        help="supported models in this implementation are CNN and LSTM.")
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing the training set.")
    parser.add_argument('--epoch', dest='epoch', type=int, default=200,
                        help="number of epochs in the training")
    parser.add_argument('--test_path', dest='test_path', type=str, default='dataset/datasets.xlsx',
                        help="address to test dataset.")
    parser.add_argument('--load', dest='load', type=str, default=None)#'models/sentiment_LSTM.pt')

    args = parser.parse_args()

    with warnings.catch_warnings():
        logging.basicConfig(filename='neural-arch.log', format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        warnings.filterwarnings("ignore")
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
              colored('\n===============NEURAL=ARCH===============', 'red'))
        main(args)
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'))
