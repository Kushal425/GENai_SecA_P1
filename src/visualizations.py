from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def plot_top_keywords(keywords):
    words, scores = zip(*keywords)
    fig, ax = plt.subplots()
    ax.barh(words, scores)
    ax.invert_yaxis()
    return fig