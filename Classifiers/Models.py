import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def trained_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model_type: str) -> object:
    if model_type == "LinearSVC":
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])
    elif model_type == "DecisionTreeClassifier":
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                             ('chi', SelectKBest(chi2, k=1000)),
                             ('clf', DecisionTreeClassifier(max_depth=5))])
    elif model_type == "AdaBoost":
        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                             ('chi', SelectKBest(chi2, k=1000)),
                             ('clf', AdaBoostClassifier(n_estimators=50))])
    else:
        print("Incorrect classifier.")
    return pipeline.fit(X_train, y_train)


def prediction(text: str, model: Pipeline):
    return model.predict([text])


def accurcy(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    y_pred = model.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred),2))


def compare_predictions(reviews: list, classifiers: list, cleaned_data) -> None:
    pretty_table = PrettyTable()
    pretty_table.add_column("Review", reviews)
    for classifier in classifiers:
        model = trained_model(cleaned_data['text'], cleaned_data["stars"], classifier)
        list = []
        for rev in reviews:
            list.append(prediction(rev, model))
        pretty_table.add_column(classifier, list)
    print(pretty_table)
