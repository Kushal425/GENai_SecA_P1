import numpy as np

def extract_keywords(tfidf_matrix, feature_names, top_n=15):
    scores = np.sum(tfidf_matrix.toarray(), axis=0)
    word_scores = list(zip(feature_names, scores))
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]