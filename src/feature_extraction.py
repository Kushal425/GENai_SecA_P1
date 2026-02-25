from sklearn.feature_extraction.text import TfidfVectorizer
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer