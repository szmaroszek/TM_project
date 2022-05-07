import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from dataOperations.cleanData import text_tokenizer


# get model
def get_linearSVC(X_data: pd.DataFrame, y_data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.3, random_state=0)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k='all')),
                         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))
                         ])
    return pipeline.fit(X_train, y_train)


# get 10 keywords
# def get_keywords(target_names: list, model: Pipeline):
def get_keywords(df: pd.DataFrame):
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

    print(top_10_count)


def get_top_10(df: pd.DataFrame):
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

    for top in top_10:
        print(names_out[top])


def get_prediction(text: str, model: Pipeline):
    print(model.predict([text]))
