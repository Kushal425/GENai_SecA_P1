from gensim.models import CoherenceModel
def calculate_coherence(lda_model, texts, dictionary):
    tokenized_texts = [text.split() for text in texts]
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()