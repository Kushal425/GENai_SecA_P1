from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

def summarize_text(text, num_sentences=5):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    ranked_sentences = sorted(
        ((score, sentence) for score, sentence in zip(sentence_scores, sentences)),
        reverse=True
    )
    summary = [sentence for score, sentence in ranked_sentences[:num_sentences]]
    return " ".join(summary)