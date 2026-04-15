import streamlit as st
import pandas as pd

# ── M1 imports ─────────────────────────────────────────────────────────────────
from src.preprocessing import preprocess_text, get_text_stats
from src.document_loader import load_uploaded_files
from src.feature_extraction import build_tfidf
from src.topic_model import build_lda_model
from src.evaluation import calculate_coherence
from src.keyword_extractor import extract_keywords
from src.summarizer import summarize_text
from src.visualizations import generate_wordcloud, plot_top_keywords, plot_topic_distribution

# ── M2 imports ─────────────────────────────────────────────────────────────────
from src.agent.graph import run_research_agent
from src.agent.report_generator import format_report
from src.pdf_export import generate_pdf_report

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchScope",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   {font-size: 2.2rem; font-weight: 800; color: #a78bfa;}
    .sub-title    {font-size: 1rem; color: #9ca3af; margin-bottom: 1.5rem;}
    .m2-title     {font-size: 2.2rem; font-weight: 800; color: #34d399;}
    .m2-sub       {font-size: 1rem; color: #9ca3af; margin-bottom: 1.5rem;}
    .keyword-badge {
        display:inline-block; background:#312e81; color:#c4b5fd;
        border-radius:6px; padding:2px 10px; margin:3px; font-size:.85rem;
    }
    .finding-item {
        background:#1f2937; border-left:4px solid #34d399;
        border-radius:6px; padding:0.6rem 1rem; margin:4px 0; color:#d1fae5;
    }
    .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    div[data-testid="stRadio"] > div {
        display: flex; gap: 12px;
    }
    div[data-testid="stRadio"] label {
        background: #1f2937; border: 1px solid #374151;
        border-radius: 8px; padding: 8px 20px;
        color: #d1d5db; font-size: 0.95rem; cursor: pointer;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #4c1d95; border-color: #7c3aed; color: white; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Mode Toggle (Top of Page) ──────────────────────────────────────────────────
st.markdown("#### 🔄 Select Mode")
mode = st.radio(
    label="",
    options=["🔬 Milestone 1: Classical NLP", "🤖 Milestone 2: Agentic AI"],
    horizontal=True,
    key="mode_toggle",
    label_visibility="collapsed",
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 1 UI
# ══════════════════════════════════════════════════════════════════════════════
if "Milestone 1" in mode:

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 ResearchScope")
        st.markdown("*Intelligent NLP Research Analyzer*")
        st.divider()

        input_mode = st.radio(
            "📥 Input Mode",
            ["📝 Paste Text", "📂 Upload Documents"],
            help="Choose how to provide your research content."
        )
        st.divider()

        st.markdown("### ⚙️ Model Settings")
        num_topics = st.slider("Number of LDA Topics", min_value=2, max_value=10, value=5)
        num_keywords = st.slider("Top Keywords to Extract", min_value=5, max_value=30, value=15)
        num_summary_sentences = st.slider("Summary Sentences", min_value=2, max_value=10, value=5)
        st.divider()

        st.markdown("### ℹ️ About")
        st.info("Milestone 1 – Traditional NLP pipeline: TF-IDF + LDA + Extractive Summarization. No LLMs used.")

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<div class="main-title">🔬 ResearchScope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent Research Topic Analyzer — Milestone 1: Traditional NLP Pipeline</div>', unsafe_allow_html=True)

    # ── Input Section ────────────────────────────────────────────────────────
    raw_text = ""
    doc_names = []

    if "Paste Text" in input_mode:
        raw_text = st.text_area(
            "✍️ Paste your research text below",
            height=200,
            placeholder="Paste any research paper abstract, article, report, or multi-paragraph text here…"
        )
        if raw_text.strip():
            doc_names = ["Pasted Text"]
    else:
        uploaded_files = st.file_uploader(
            "📂 Upload research documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload one or more PDF or TXT documents. They will be combined for analysis."
        )
        if uploaded_files:
            with st.spinner("Reading documents…"):
                docs = load_uploaded_files(uploaded_files)
            if docs:
                raw_text = "\n\n".join(d["text"] for d in docs)
                doc_names = [d["name"] for d in docs]
                st.success(f"✅ Loaded {len(docs)} document(s): {', '.join(doc_names)}")
            else:
                st.error("Could not extract text from the uploaded files.")

    # ── Analyse Button ───────────────────────────────────────────────────────
    run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if run_analysis:
        if not raw_text.strip():
            st.warning("⚠️ Please provide some text or upload a document first.")
        else:
            with st.spinner("Running NLP pipeline…"):
                stats = get_text_stats(raw_text)
                processed = preprocess_text(raw_text)
                corpus = [processed]
                tfidf_matrix, feature_names, _ = build_tfidf(corpus)
                keywords = extract_keywords(tfidf_matrix, feature_names, top_n=num_keywords)
                lda_model, topics, corpus_g, dictionary = build_lda_model(corpus, num_topics=num_topics)
                try:
                    coherence = calculate_coherence(lda_model, corpus, dictionary)
                except Exception:
                    coherence = None
                summary, scored_sentences = summarize_text(raw_text, num_sentences=num_summary_sentences)

            # ── Metrics Row ──────────────────────────────────────────────────
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Documents", len(doc_names))
            col2.metric("🔤 Words", f"{stats['words']:,}")
            col3.metric("📃 Sentences", stats["sentences"])
            col4.metric("🔠 Unique Tokens", stats["unique_tokens"])
            st.divider()

            # ── Tabs ─────────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Keywords", "🧩 Topics", "📝 Summary", "📈 Visualizations"
            ])

            with tab1:
                st.markdown("### 🔑 Top Keywords by TF-IDF Score")
                st.caption("Words ranked by their term frequency–inverse document frequency weight.")
                badge_html = "".join(
                    f'<span class="keyword-badge">{w} <b>({s:.3f})</b></span>'
                    for w, s in keywords
                )
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("---")
                df_kw = pd.DataFrame(keywords, columns=["Keyword", "TF-IDF Score"])
                df_kw["TF-IDF Score"] = df_kw["TF-IDF Score"].round(4)
                df_kw.index += 1
                st.dataframe(df_kw, use_container_width=True)

            with tab2:
                st.markdown("### 🧩 LDA Topic Clusters")
                if coherence is not None:
                    coh_col1, coh_col2 = st.columns([1, 3])
                    coh_col1.metric("Coherence Score (Cᵥ)", f"{coherence:.4f}",
                                    help="Higher is better. Scores 0.4–0.7 indicate good topic separation.")
                st.caption("Each topic is represented by its highest-probability words.")
                st.markdown("---")
                for topic_id, topic_str in topics:
                    with st.expander(f"🔹 Topic {topic_id + 1}", expanded=(topic_id < 3)):
                        pairs = []
                        for part in topic_str.split(" + "):
                            try:
                                weight, word = part.split('*"')
                                pairs.append((word.strip('"'), float(weight)))
                            except ValueError:
                                continue
                        badges = "".join(
                            f'<span class="keyword-badge">{w} ({wt:.3f})</span>'
                            for w, wt in pairs
                        )
                        st.markdown(badges, unsafe_allow_html=True)

            with tab3:
                st.markdown("### 📝 Extractive Summary")
                st.caption("Sentences selected by TF-IDF score, in their original document order.")
                st.markdown("---")
                if summary:
                    st.success(summary)
                else:
                    st.warning("Not enough text to generate a summary.")
                if scored_sentences:
                    st.markdown("#### Sentence Scores")
                    df_sum = pd.DataFrame(scored_sentences, columns=["Sentence", "Score"])
                    df_sum["Score"] = df_sum["Score"].round(4)
                    df_sum.index += 1
                    st.dataframe(df_sum, use_container_width=True)

            with tab4:
                st.markdown("### 📈 Analytical Visualizations")
                v1, v2 = st.columns(2)
                with v1:
                    st.markdown("#### ☁️ Word Cloud")
                    if processed.strip():
                        st.pyplot(generate_wordcloud(processed))
                    else:
                        st.info("Not enough text for a word cloud.")
                with v2:
                    st.markdown("#### 📊 Keyword Bar Chart")
                    if keywords:
                        st.pyplot(plot_top_keywords(keywords))
                st.markdown("---")
                st.markdown("#### 🧩 Topic Word Distribution")
                topic_fig = plot_topic_distribution(topics)
                if topic_fig:
                    st.pyplot(topic_fig)


# ══════════════════════════════════════════════════════════════════════════════
# MILESTONE 2 UI
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 2" in mode:

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 ResearchScope")
        st.markdown("*Agentic AI Research Assistant*")
        st.divider()
        st.markdown("### ℹ️ How it works")
        st.info(
            "Milestone 2 — Agentic AI pipeline:\n\n"
            "1. 🔎 Search the web (DuckDuckGo)\n"
            "2. 📥 Retrieve source content\n"
            "3. ✅ Validate & filter sources\n"
            "4. 🧠 Summarize with LLM (Groq)\n"
            "5. 📝 Generate structured report"
        )
        st.divider()
        st.markdown("**LLM:** Groq — `llama3-8b-8192`")
        st.markdown("**Search:** DuckDuckGo (no API key)")
        st.markdown("**Extension:** PDF Export")

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<div class="m2-title">🤖 Agentic Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="m2-sub">Milestone 2 — LangGraph · Web Search · LLM Summarization · Structured Reports</div>', unsafe_allow_html=True)

    # ── Query Input ──────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Enter your research query",
        placeholder="e.g. What are the latest advances in transformer-based NLP models?",
    )

    run_agent = st.button("🚀 Research", type="primary", use_container_width=True)

    if run_agent:
        if not query.strip():
            st.warning("⚠️ Please enter a research query first.")
        else:
            # ── Live Status Indicators ───────────────────────────────────────
            final_state = None
            with st.status("🤖 Running Research Agent...", expanded=True) as status:
                st.write("🔎 **Step 1/5** — Searching the web...")
                st.write("📥 **Step 2/5** — Retrieving source content...")
                st.write("✅ **Step 3/5** — Validating sources...")
                st.write("🧠 **Step 4/5** — Summarizing with LLM...")
                st.write("📝 **Step 5/5** — Generating structured report...")
                final_state = run_research_agent(query)
                if final_state.get("error"):
                    status.update(label=f"⚠️ Completed with warnings", state="error")
                else:
                    status.update(label="✅ Research complete!", state="complete")

            # ── Error Warning ────────────────────────────────────────────────
            if final_state.get("error"):
                st.warning(f"⚠️ Note: {final_state['error']}")

            # ── Format Report ─────────────────────────────────────────────────
            report = format_report(final_state)

            # ── Result Tabs ───────────────────────────────────────────────────
            r_tab1, r_tab2, r_tab3 = st.tabs([
                "📄 Research Report", "🔗 Sources", "🧠 Raw LLM Summary"
            ])

            with r_tab1:
                st.markdown(f"## {report['title']}")
                st.divider()

                st.markdown("### Abstract")
                st.info(report["abstract"])

                st.markdown("### Key Findings")
                findings = report["key_findings"]
                if findings:
                    for finding in findings:
                        st.markdown(
                            f'<div class="finding-item">✦ {finding}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No key findings were extracted.")

                st.markdown("### Conclusion")
                st.success(report["conclusion"])

                # ── PDF Export Button ────────────────────────────────────────
                st.divider()
                try:
                    pdf_bytes = generate_pdf_report(report)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_bytes,
                        file_name=f"ResearchScope_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

            with r_tab2:
                st.markdown("### 🔗 Validated Sources")
                sources = report["sources"]
                if sources:
                    for i, source in enumerate(sources, 1):
                        title = source.get("title", "Unknown Source")
                        url = source.get("url", "")
                        col_a, col_b = st.columns([3, 2])
                        col_a.markdown(f"**{i}. {title}**")
                        if url:
                            col_b.markdown(f"[Open link ↗]({url})")
                        st.divider()
                else:
                    st.warning("No validated sources found.")

            with r_tab3:
                st.markdown("### 🧠 Raw LLM Summary")
                st.caption("This is the intermediate summary generated by the LLM before structuring the report.")
                st.markdown("---")
                raw_summary = final_state.get("llm_summary", "")
                if raw_summary:
                    st.markdown(raw_summary)
                else:
                    st.warning("No LLM summary available.")