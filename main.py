from matplotlib import pyplot as plt
from nltk import collections
from wordcloud import WordCloud

from dataOperations.getData import get_data
from dataOperations.cleanData import clean_data
from dataOperations.wordCloud import generate_cloud
from Classifiers.Models import get_linearSVC, get_keywords, get_prediction, get_top_10


# sample review data
data = get_data('data/sample_data.csv')

data['cleaned'] = clean_data(data)

model = get_linearSVC(data['cleaned'], data["stars"])

get_keywords(data)
get_top_10(data)

# predictions
get_prediction('That was an awesome place. Great food! Highly recommend', model)
get_prediction('it was good but lacking, maybe we will visit again', model)
get_prediction('food was cold and awful, wont come back', model)
get_prediction('The worst restaurant I have ever been!!', model)
get_prediction('Portions were alright, but we waited for a long time and there was a lot of people', model)

generate_cloud(data['cleaned'])