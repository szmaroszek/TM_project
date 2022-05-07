from prettytable import PrettyTable
import matplotlib.pyplot as plt


def get_graph_count(data: dict) -> None:
    x = PrettyTable()
    x.field_names = ["Word", "Count"]
    for word, count in data.items():
        x.add_row([word, count])

    plt.suptitle('Most common words in restaurant reviews')
    plt.bar(data.keys(), data.values())
    plt.show()
