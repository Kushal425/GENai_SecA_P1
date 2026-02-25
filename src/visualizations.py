from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "text.color": "#ccc",
    "grid.color": "#333",
})

def generate_wordcloud(text):
    """Generate a styled word cloud figure."""
    wc = WordCloud(
        width=900,
        height=400,
        background_color="#0e1117",
        colormap="plasma",
        max_words=100,
        prefer_horizontal=0.9,
        collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud", fontsize=14, pad=10, color="#ccc")
    fig.tight_layout()
    return fig

def plot_top_keywords(keywords):
    """Horizontal bar chart for top TF-IDF keywords."""
    if not keywords:
        return None
    words, scores = zip(*keywords)
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(words)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(words, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("TF-IDF Score")
    ax.set_title("Top Keywords by TF-IDF Score", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_topic_distribution(topics):
    """
    Bar chart showing top words per LDA topic.
    topics: list of (topic_id, topic_string) from lda_model.print_topics()
    """
    if not topics:
        return None

    fig, axes = plt.subplots(1, len(topics), figsize=(4 * len(topics), 4), sharey=False)
    if len(topics) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, (topic_id, topic_str) in zip(axes, topics):
        # Parse "0.123*\"word\" + ..." format
        pairs = []
        for part in topic_str.split(" + "):
            try:
                weight, word = part.split('*"')
                word = word.strip('"')
                pairs.append((word, float(weight)))
            except ValueError:
                continue
        if not pairs:
            continue
        words, weights = zip(*pairs[:8])
        ax.barh(words, weights, color=colors[topic_id % len(colors)])
        ax.invert_yaxis()
        ax.set_title(f"Topic {topic_id + 1}", fontsize=11)
        ax.set_xlabel("Weight")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("LDA Topic Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig