from matplotlib import pyplot as plt
from nltk import collections
from wordcloud import WordCloud

from dataOperations.getData import get_data
from dataOperations.cleanData import clean_data
from dataOperations.wordCloud import generate_cloud
from Classifiers.Models import get_linearSVC, get_prediction
from dataOperations.getKeywords import get_top_10_keywords, get_keywords_count
from dataOperations.getGraph import get_graph_count


# sample review data
data = get_data('data/sample_data.csv')

print('10 most important keywords: ' + ', '.join(get_top_10_keywords(data)))
get_graph_count(get_keywords_count(data))

data_cleaned = clean_data(data)

linearSVC_model = get_linearSVC(data_cleaned['text'], data_cleaned["stars"])


# predictions
get_prediction('That was an awesome place. Great food! Highly recommend', linearSVC_model)
get_prediction('it was good but lacking, maybe we will visit again', linearSVC_model)
get_prediction('food was cold and awful, wont come back', linearSVC_model)
get_prediction('The worst restaurant I have ever been!!', linearSVC_model)
get_prediction('Portions were alright, but we waited for a long time and there was a lot of people', linearSVC_model)

generate_cloud(data_cleaned['text'])
