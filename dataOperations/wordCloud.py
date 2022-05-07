from matplotlib import pyplot as plt
from nltk import collections
from wordcloud import WordCloud


def generate_cloud(data_df) -> None:
    data_dict = collections.Counter([y for x in data_df.values.flatten() for y in x.split()])
    wc = WordCloud()
    wc.generate_from_frequencies(data_dict)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
