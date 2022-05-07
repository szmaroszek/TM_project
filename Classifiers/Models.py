import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# get linearSVC model
def get_linearSVC(X_data: pd.DataFrame, y_data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.3, random_state=0)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k='all')),
                         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))
                         ])
    return pipeline.fit(X_train, y_train)


def get_prediction(text: str, model: Pipeline):
    print(model.predict([text]))
