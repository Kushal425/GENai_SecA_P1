from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

def summarize_text(text, num_sentences=5):
    """
    Extractive summarization using TF-IDF sentence scoring.
    Preserves the original sentence order in the output.
    Returns: (summary_string, list of (sentence, score) tuples)
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return "", []

    # Cap num_sentences to available sentences
    num_sentences = min(num_sentences, len(sentences))

    if len(sentences) == 1:
        return sentences[0], [(sentences[0], 1.0)]

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    except ValueError:
        # Fallback: return first N sentences if TF-IDF fails
        return " ".join(sentences[:num_sentences]), []

    # Rank sentences by score, keep original order for readability
    ranked_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
    selected_indices = sorted(ranked_indices)  # preserve document order

    summary_sentences = [sentences[i] for i in selected_indices]
    scored_sentences = [(sentences[i], float(sentence_scores[i])) for i in selected_indices]

    return " ".join(summary_sentences), scored_sentences