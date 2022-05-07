from matplotlib import pyplot as plt
from nltk import collections
from prettytable import PrettyTable
from wordcloud import WordCloud
from dataOperations.getData import get_data
from dataOperations.cleanData import clean_data
from dataOperations.wordCloud import generate_cloud
from dataOperations.getKeywords import get_top_10_keywords, get_keywords_count
from dataOperations.getGraph import get_graph_count
from Classifiers.Models import trained_model, prediction, accurcy, compare_predictions

# get sample data
data = get_data('data/sample_data.csv')

# top 10 most important keywords
print('10 most important keywords: ' + ', '.join(get_top_10_keywords(data)))
get_graph_count(get_keywords_count(data))

# prepare data
cleaned_data = clean_data(data)


# show word cloud
generate_cloud(cleaned_data['text'])


# train model
classifiers = ["LinearSVC", "DecisionTreeClassifier", "AdaBoost"]
for classifier in classifiers:
    model = trained_model(cleaned_data['text'], cleaned_data["stars"], classifier)
    accurcy(model, cleaned_data['text'], cleaned_data["stars"])


reviews = ["That was an awesome place. Great food! Highly recommend",
           "It was good but lacking, maybe we will visit again",
           "Food was cold and awful, wont come back",
           "The worst restaurant I have ever been!!",
           "Portions were alright, but we waited for a long time and there was a lot of people"]


compare_predictions(reviews, classifiers, cleaned_data)