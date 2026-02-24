import gensim
from gensim import corpora

def build_lda_model(texts, num_topics=5):
    tokenized_texts = [text.split() for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = gensim.models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10
    )

    topics = lda_model.print_topics()
    return lda_model, topics, corpus, dictionary