import streamlit as st
from src.preprocessing import preprocess_text
from src.feature_extraction import build_tfidf
from src.topic_model import build_lda_model
from src.evaluation import calculate_coherence
from src.keyword_extractor import extract_keywords
from src.summarizer import summarize_text
from src.visualizations import generate_wordcloud, plot_top_keywords

st.title("ResearchScope – Intelligent Topic Analyzer")

text_input = st.text_area("Enter Research Text")

if st.button("Analyze"):
    processed = preprocess_text(text_input)

    corpus = [processed]

    tfidf_matrix, feature_names, _ = build_tfidf(corpus)
    keywords = extract_keywords(tfidf_matrix, feature_names)

    lda_model, topics, corpus_g, dictionary = build_lda_model(corpus)
    coherence = calculate_coherence(lda_model, corpus, dictionary)

    summary = summarize_text(text_input)

    tab1, tab2, tab3, tab4 = st.tabs(["Keywords", "Topics", "Summary", "Visualizations"])

    with tab1:
        st.write(keywords)

    with tab2:
        st.write(topics)
        st.write("Coherence Score:", coherence)

    with tab3:
        st.write(summary)

    with tab4:
        st.pyplot(generate_wordcloud(processed))
        st.pyplot(plot_top_keywords(keywords))