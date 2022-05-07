import pandas as pd
import re
from nltk import SnowballStemmer, word_tokenize, PorterStemmer
from nltk.corpus import stopwords


def clean_data(df: pd.DataFrame):
    for index, row in df.iterrows():
        text_tokenizer(row[2])
        df.at[index,'text'] = ' '.join(text_tokenizer(row[2]))
    return df


def text_tokenizer(text):
    text = clear_text(text)
    text = stem_text(text)
    text = clear_stopwords(text)
    text = clear_short_words(text)
    return text.split()


def clear_text(text):
    text = re.sub(r'[.,]', '', text)
    text = re.sub(r'[:;][^\s]{1,2}', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[.,]', '', text)
    text = " ".join(text.split())
    return text


def clear_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = ''
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence += ' ' + w
    return filtered_sentence


def stem_text(text: str) -> str:
    porter = PorterStemmer()
    return " ".join(porter.stem(word) for word in text.split())


def clear_short_words(text):
    return " ".join(word for word in text.split() if len(word) > 3)
    