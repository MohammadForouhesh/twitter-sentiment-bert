from src.params import sw_persian
import pandas as pd
import re


def correction(series:pd.Series):
    assert isinstance(series, pd.Series)
    
    for line in series:
        line = line.replace('\n', '').replace('.', '')
        line = line.split(' ')

        yield list(map(int, line))


def remove_emoji(text:str) -> str:
    assert isinstance(text, str)
    
    emoji_pattern = re.compile(pattern="["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001F300-\U0001F5FF"
                                       u"\U0001F1E6-\U0001F1FF"
                                       u"\U00002700-\U000027BF"
                                       u"\U0001F900-\U0001F9FF"
                                       u"\U0001F600-\U0001F64F"
                                       u"\U0001F680-\U0001F6FF"
                                       u"\U00002600-\U000026FF"
                                       "]+", flags=re.UNICODE)
    
    return str(emoji_pattern.sub(r'', text))


def remove_redundent_characters(text:str) -> str:
    assert isinstance(text, str)
    
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  # Removed @mentions
    text = re.sub(r'_[A-Za-z0-9]+', ' ', text)  # Removed underlines
    text = re.sub(r'/(\r\n)+|\r+|\n+|\t+/', ' ', text)  # Removed \n
    text = re.sub(r'#', ' ', text)  # Removing the '#' symbol
    text = re.sub(r'RT[\s]+', ' ', text)  # Removing RT
    text = re.sub(r'https?:\/\/\S+', ' ', text)  # Remove the hyper link
    text = re.sub(r'\([ا-ی]{1,3}\)', ' ', text)  # Remove abbreviations
    text = re.sub(r"[\(\)]", " ", text)  # remove parantesis
    text = re.sub(r"\d|[۰-۹]", " ", text)
    text = re.sub(r"&|:", " ", text)
    text = re.sub(r"[A-Za-z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"@|=", " ", text)
    text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = re.sub(r"{|}|;|\[|\]|\||؟|!|\+|\-|\*|\$", " ", text)
    text = re.sub(r"¹²|\/", " ", text)
    text = re.sub(r"»|>|<|«|,|؛|،|%|؟", " ", text)
    text = re.sub("\.|\^|,", " ", text)
    return text


def remove_stop_words(text:str) -> str:
    assert isinstance(text, str)
    
    return ' '.join([word for word in text.split(' ') if word not in sw_persian])


def preprocess(sentence:str) -> str:
    return remove_stop_words(
            remove_redundent_characters(
                remove_emoji(sentence)))