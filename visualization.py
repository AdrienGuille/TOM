# coding: utf-8
from bokeh.plotting import ColumnDataSource, figure, show, output_file
from bokeh.models import HoverTool
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import seaborn

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"


def plot_topic_distribution(distribution, file_path=None):
    data_x = range(0, len(distribution))
    plt.clf()
    plt.xticks(np.arange(0, len(distribution), 1.0))
    plt.bar(data_x, distribution, align='center')
    plt.title('Topic distribution')
    plt.ylabel('probability')
    plt.xlabel('topic')
    if file_path is None:
        file_path = 'output/topic_distribution.png'
    plt.savefig(file_path)


def plot_word_distribution(distribution, file_path=None):
    data_x = []
    data_y = []
    for weighted_word in distribution:
        data_x.append(weighted_word[1])
        data_y.append(weighted_word[0])
    plt.clf()
    plt.bar(range(len(data_x)), data_y, align='center')
    plt.xticks(range(len(data_x)), data_x, size='small', rotation='vertical')
    plt.title('Word distribution')
    plt.ylabel('probability')
    plt.xlabel('word')
    if file_path is None:
        file_path = 'output/word_distribution.png'
    plt.savefig(file_path)


def plot_topic_evolution(series, file_path=None):
    data_x = range(0, len(series))
    plt.clf()
    plt.xticks(np.arange(0, len(series), 1.0))
    plt.plot(data_x, series)
    plt.title('Topic popularity')
    plt.ylabel('frequency')
    plt.xlabel('year')
    if file_path is None:
        file_path = 'output/topic_evolution.png'
    plt.savefig(file_path)


def heat_map(document_topic_matrix):
    colormap = [
        "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#6a3d9a", "#6a3d9a"
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#6a3d9a", "#6a3d9a", "#6a3d9a"
    ]
    topic = []
    document = []
    color = []
    probability = []
    alphas = []
    doc_count = 0
    for doc in document_topic_matrix:
        topic_count = 0
        most_likely_topic = doc.index(max(doc))
        for prob in doc:
            topic.append(topic_count)
            document.append(doc_count)
            probability.append(prob)
            alphas.append(prob)
            if most_likely_topic == topic_count:
                color.append(colormap[topic_count])
            else:
                color.append('lightgrey')
            topic_count += 1
        if doc_count > 50:
            break
        doc_count += 1
    source = ColumnDataSource(
        data=dict(topic=topic, document=document, color=color, alpha=alphas, probability=probability)
    )
    output_file("output/topic_distribution.html")
    p = figure(title="Les Mis Occurrences",
        x_axis_location="above", tools="resize,hover,save",
        x_range=[str(i) for i in range(50)], y_range=list(reversed([str(i) for i in range(15)])))
    p.plot_width = 800
    p.plot_height = 400
    p.rect('document', 'topic', 0.9, 0.9, source=source,
         color='color', alpha='alphas', line_color=None)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ('document, topic', '@document, @topic'),
        ('probability', '@probability'),
    ])
    show(p)