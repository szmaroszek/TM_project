import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from dataOperations.cleanData import text_tokenizer


def get_keywords_count(df: pd.DataFrame) -> dict:
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    X_transform = vectorizer.fit_transform(df['text'])
    x_transformed_arr = np.sum(X_transform.toarray(), 0)
    top_10 = []
    count = []
    top_10_count = {}
    for i in range(10):
        arr_index = (np.argmax(x_transformed_arr, 0))
        top_10.append(arr_index)
        count.append(x_transformed_arr[arr_index])
        x_transformed_arr[arr_index] = 0
        i += 1

    names_out = vectorizer.get_feature_names_out()

    for i, top in enumerate(top_10):
        top_10_count[names_out[top]] = count[i]

    return top_10_count


def get_top_10_keywords(df: pd.DataFrame) -> list:
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    X_transform = vectorizer.fit_transform(df['text'])
    x_transformed_arr = np.sum(X_transform.toarray(), 0)
    top_10 = []
    for i in range(10):
        arr_index = (np.argmax(x_transformed_arr, 0))
        top_10.append(arr_index)
        x_transformed_arr[arr_index] = 0
        i += 1

    names_out = vectorizer.get_feature_names_out()

    return [names_out[top] for top in top_10]
