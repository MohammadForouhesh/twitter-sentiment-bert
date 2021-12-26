from src.preprocessing.Preprocessing import correction, preprocess
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src.metrics import ir_metrics
from src.model import LSTM, CNN
from src.running.Run import run
from src.params import device
from termcolor import colored
from datetime import datetime
from torch import nn
import pandas as pd
import argparse
import warnings
import logging
import torch
import gc

gc.enable()
warnings.filterwarnings("ignore", category=SyntaxWarning)


emb_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def preparation(args) -> (pd.DataFrame, pd.DataFrame):
    train_df = pd.read_csv(args.train_path)
    
    test_df = pd.read_csv(args.test_path)
    df = train_df.append(test_df)
    df['category_id'] = df['label'].factorize()[0]
    
    category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    
    df.label = list(correction(df.label))
    df['preprocessed'] = df.text.apply(preprocess)
    
    return df[:5999], df[5999:], category_to_id


def main():
    parser = argparse.ArgumentParser(description='Exa Emotion Detection')
    parser.add_argument('--train_path', dest='train_path', type=str, default='dataset/Emotion.csv',
                        help='Raw dataset file address.')
    parser.add_argument('--augment', dest='augment', type=int, default=None,
                        help='augment the dataset to learn better.')
    parser.add_argument('--model_name', dest='model_name', type=str, default='lstm',
                        help="supported models in this implementation are CNN and LSTM.")
    parser.add_argument('--preprocess', dest='preprocess', type=bool, default=True,
                        help="whether or not preprocessing the training set.")
    parser.add_argument('--epoch', dest='epoch', type=int, default=100,
                        help="number of epochs in the training")
    parser.add_argument('--test_path', dest='test_path', type=str, default='dataset/EmotionTest.csv',
                        help="address to test dataset.")
    
    args = parser.parse_args()

    train_df, test_df, category_to_id = preparation(args)

    sentences = list(train_df.preprocessed)
    sentence_embeddings = emb_model.encode(sentences)

    inputs = torch.from_numpy(sentence_embeddings).to(device)
    target = torch.cuda.LongTensor(train_df.category_id)

    train_ds = TensorDataset(inputs, target)

    sentences = list(test_df.preprocessed)
    sentence_embeddings = emb_model.encode(sentences)

    inputs = torch.from_numpy(sentence_embeddings).to(device)
    target = torch.cuda.LongTensor(test_df.category_id)

    test_ds = TensorDataset(inputs, target)

    bach_size = 5 if args.model_name == 'lstm' else 1

    train_dl = DataLoader(train_ds, bach_size, shuffle=True)
    test_dl = DataLoader(test_ds, bach_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n====================TRAIN=' + args.model_name + '=====================', 'red'))
    if args.model_name == 'lstm':
        model = LSTM(output_size=len(category_to_id)).to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    
        trained_model = run(model=model, iterator=train_dl, optimizer=optimizer,
                            loss_function=loss_function, n_epoch=epoch, if_lstm=True)

    elif args.model_name == 'cnn':
    
        model = CNN(output_dim=len(category_to_id)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    
        trained_model = run(model=model, iterator=train_dl, optimizer=optimizer,
                            loss_function=loss_function)

    ir_metrics(model=trained_model, iterator=train_dl)

    print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
          colored('\n====================Test==' + args.model_name + '=====================', 'red'))

    ir_metrics(model=trained_model, iterator=test_dl)
        
        
if __name__ == '__main__':
    with warnings.catch_warnings():
        logging.basicConfig(filename='exa_model.log', format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        warnings.filterwarnings("ignore")
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'),
              colored('\n===============EXA=EmotionDetection===============', 'red'))
        main()
        print(colored('[' + str(datetime.now().hour) + ':' + str(datetime.now().minute) + ']', 'cyan'))
