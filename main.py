from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

from twitter_sentiment.model.Cnn import CNN2d1, CNN2d
from twitter_sentiment.model.Gcnn import GCNN
from twitter_sentiment.model.Sentiment import SentimentModel
from twitter_sentiment.params import MAX_LEN, TRAIN_BATCH_SIZE, LABEL_LIST, id2label, device, MODEL_NAME_OR_PATH, label2id, \
    EPOCHS, LEARNING_RATE
from twitter_sentiment.preprocessing.Dataset import create_data_loader
from twitter_sentiment.preprocessing.Preprocessing import preprocess, remove_redundent_characters
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from twitter_sentiment.model.Lr import LinearRegression
from torch.utils.data import DataLoader
from twitter_sentiment.metrics import ir_metrics
from twitter_sentiment.model import LSTM, CNN
from twitter_sentiment.running.Run import run
from termcolor import colored
from datetime import datetime
from tqdm import tqdm
from torch import nn
import pandas as pd
import numpy as np
import argparse
import warnings
import logging
import pickle
import torch
import gc

gc.enable()
warnings.filterwarnings("ignore", category=SyntaxWarning)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def preparation(dataframe) -> (np.ndarray, np.ndarray):
    dataframe.fillna('', inplace=True)
    dataframe['text'] = dataframe.text.apply(remove_redundent_characters)

    dataframe.replace('', float('NaN'))
    dataframe.dropna(inplace=True)
    text = dataframe.text
    label = dataframe.sentiment
    return list(text), list(label)


def inference(model, tokenizer, sentence):
    input = create_data_loader(tokenizer=tokenizer, tweets=[sentence], targets=None, max_len=MAX_LEN,
                               label_list=LABEL_LIST, batch_size=1)
    return id2label[int(model(input)[0].argmax())]


def main(args):
    df = pd.read_excel(args.train_path)
    # df = df_normalizer(df)
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=list(df.sentiment))
    train_df, eval_df = train_test_split(train_df, test_size=0.1, stratify=list(train_df.sentiment))

    X_train, y_train = preparation(train_df)
    X_eval, y_eval = preparation(eval_df)
    X_test, y_test = preparation(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    embedding = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
    config = AutoConfig.from_pretrained(
        MODEL_NAME_OR_PATH, **{
            'label2id': label2id,
            'id2label': id2label,
        })

    train_dl = create_data_loader(tokenizer=tokenizer, tweets=X_train, targets=y_train, label_list=LABEL_LIST,
                                  max_len=MAX_LEN, batch_size=TRAIN_BATCH_SIZE, device=device)
    eval_dl  = create_data_loader(tokenizer=tokenizer, tweets=X_eval, targets=y_eval, label_list=LABEL_LIST,
                                  max_len=MAX_LEN, batch_size=TRAIN_BATCH_SIZE, device=device)
    test_dl  = create_data_loader(tokenizer=tokenizer, tweets=X_test, targets=y_test, label_list=LABEL_LIST,
                                  max_len=MAX_LEN, batch_size=TRAIN_BATCH_SIZE, device=device)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n===============TRAIN=' + args.model_name.upper() + '===============', 'red'))
    model = SentimentModel(embedding, config).to(device)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, lr=LEARNING_RATE)
    total_steps = len(train_dl) * args.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    trained_model = run(model=model, train_iterator=train_dl, eval_iterator=eval_dl, optimizer=optimizer,
                        scheduler=scheduler, loss_function=loss_function, n_epoch=args.epoch)

    ir_metrics(model=trained_model, iterator=train_dl)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n===============Test==' + args.model_name.upper() + '===============', 'red'))

    ir_metrics(model=trained_model, iterator=test_dl)


def df_normalizer(df):
    negative_data = df[df['sentiment'] == 'sad']
    neutral_data = df[df['sentiment'] == 'meh']
    positive_data = df[df['sentiment'] == 'happy']
    cutting_point = min(len(negative_data), len(neutral_data), len(positive_data))
    print(cutting_point)
    if cutting_point <= len(negative_data):
        negative_data = negative_data.sample(n=cutting_point).reset_index(drop=True)
    if cutting_point <= len(neutral_data):
        neutral_data = neutral_data.sample(n=cutting_point).reset_index(drop=True)
    if cutting_point <= len(positive_data):
        positive_data = positive_data.sample(n=cutting_point).reset_index(drop=True)
    df = pd.concat([negative_data, neutral_data, positive_data])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='twitter-sentiment-bert')
    parser.add_argument('--train_path', dest='train_path', type=str, default='dataset/sentiment-multilingual-approach.xlsx',
                        help='Raw dataset file address.')
    parser.add_argument('--augment', dest='augment', type=bool, default=True,
                        help='augment the dataset to learn better.')
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing the training set.")
    parser.add_argument('--epoch', dest='epoch', type=int, default=15,
                        help="number of epochs in the training")
    parser.add_argument('--test_path', dest='test_path', type=str, default='dataset/datasets.xlsx',
                        help="address to test dataset.")
    parser.add_argument('--load', dest='load', type=bool, default=True)#'models/sentiment_LSTM.pt')

    args = parser.parse_args()

    with warnings.catch_warnings():
        logging.basicConfig(filename='twitter-sentiment-bert.log', format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        warnings.filterwarnings("ignore")
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
              colored('\n============TWITTER=SENTIMENT============', 'red'))
        main(args)
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'))
