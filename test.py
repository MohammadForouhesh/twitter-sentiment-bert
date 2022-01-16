import pandas as pd
from tqdm import tqdm
import time

from transformers import pipeline


sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
df = pd.read_excel('df-3.xlsx', header=0)
en = pd.read_excel('df-2.xlsx', header=0)
en.fillna('', inplace=True)
tqdm.pandas()
print(en.columns)
df['label'] = en[' text'].progress_apply(lambda item: sentiment_analysis(item)[0]['label'])

df.to_excel('sentiment.xlsx')

df.polarity.plot(kind='kde')