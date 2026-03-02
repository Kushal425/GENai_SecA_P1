v# ResearchScope – Intelligent Topic Analyzer
### From Statistical NLP to Agentic AI Research Analysis

## Project Overview

This project focuses on the development of a multi-phase intelligent system for analyzing research documents. It transitions from a statistical, classical NLP approach in Milestone 1 to an autonomous, reasoning-based AI agent in Milestone 2.

- *Milestone 1:* Implementation of a traditional NLP pipeline using TF-IDF, LDA Topic Modeling, and extractive summarization to analyze and surface key themes from research documents — strictly utilizing classical machine learning approaches (no LLMs).
- *Milestone 2:* Extension into an agentic AI system that performs semantic reasoning, dynamic topic discovery, and intelligent summarization using LLMs and Retrieval-Augmented Generation (RAG).

---

## Constraints & Requirements

| Field | Details |
|---|---|
| *Team Members* | Kushal Sarkar, Chinmay Soni, Lakshya Bapna |
| *API Budget* | Free Tier Only (NLTK, Scikit-learn, Gensim, Streamlit) |
| *Framework* | Classical NLP (M1) / Agentic AI (M2) |
| *Hosting* | Streamlit Cloud |

---

## Technology Stack

| Component | Technology |
|---|---|
| NLP Pipeline | NLTK, spaCy (en_core_web_sm), TF-IDF |
| Topic Modeling | Gensim (LDA), CoherenceModel |
| ML / Feature Extraction | Scikit-learn (TfidfVectorizer) |
| Summarization | NLTK + Scikit-learn (Extractive) |
| UI Framework | Streamlit |
| Visualizations | Matplotlib, WordCloud, Pandas |
| Execution | Python 3.x |

---

## Milestones & Deliverables

### ✅ Milestone 1: Classical NLP Research Analysis System (Current)

*Objective:* Build a robust baseline system that identifies key themes, extracts significant vocabulary, and produces structured analytical summaries using purely statistical and classical machine learning methods.

*Key Deliverables:*

- *Document Intake:* Supports direct text pasting and multi-file uploading (.txt and .pdf) via PyPDF2. Documents are concatenated into a single corpus.
- *Dynamic Metrics Row:* Real-time statistics including document count, total word count, sentence count, and unique vocabulary size.
- *Keyword Extraction:* Identifies the most significant vocabulary using TF-IDF (Term Frequency-Inverse Document Frequency).
- *Topic Modeling:* Discovers underlying thematic clusters using LDA (Latent Dirichlet Allocation), with a Coherence Score (Cᵥ) to quantify model quality.
- *Extractive Summarization:* Generates concise summaries by scoring sentences with TF-IDF, retaining the most critical information.
- *Analytical Visualizations:* Word Cloud, Horizontal Bar Chart (TF-IDF scores), and Grouped Bar Chart (LDA topic word distributions).
- *Working Application:* Streamlit web UI running locally with a clean dark-mode interface.

---

### 🔜 Milestone 2: Agentic AI Research Assistant (Upcoming)

*Objective:* Transform the system into an autonomous agent capable of semantic reasoning, context-aware summarization, and evidence retrieval across research documents.

*Key Deliverables:*

- *Publicly Deployed App:* Hosted on Streamlit Cloud.
- *Agent Workflow:* Implementation of reasoning loops (Plan → Retrieve → Synthesize).
- *RAG Integration:* Retrieval-Augmented Generation for grounded, context-aware topic discovery.
- *Semantic Understanding:* LLM-powered summarization that understands synonyms, context, and nuance.
- *Demo Video:* Walkthrough of the agentic analysis process.

---

## Evaluation Criteria

| Phase | Weight | Criteria |
|---|---|---|
| Milestone 1 | 25% | Preprocessing Quality, TF-IDF Feature Engineering, LDA Coherence Score, UI Functional Usability |
| Milestone 2 | 30% | Agentic Reasoning Quality, RAG Implementation, Semantic Accuracy, Successful Deployment |

---

## Technical Architecture (Milestone 1)

The system follows a linear data pipeline through modular components:


Input (Text / PDF)
      ↓
Text Preprocessing     [src/preprocessing.py]
  - Lowercasing, Punctuation Removal
  - Tokenization, Stopword Removal, Lemmatization
      ↓
Feature Extraction     [src/feature_extraction.py]
  - TF-IDF Vectorization (scikit-learn)
      ↓
Topic Modeling         [src/topic_model.py]
  - LDA Model (gensim)
  - Coherence Evaluation [src/evaluation.py]
      ↓
Extractive Summarization  [src/summarizer.py]
  - Sentence Scoring & Ranking
      ↓
UI & Visualizations    [app.py, src/visualizations.py]
  - Streamlit Dashboard, Word Cloud, Charts


---

## Limitations of Current Approach

> These limitations justify the future transition to the Agentic AI system in Milestone 2.

- *Lack of Semantic Understanding:* TF-IDF and LDA rely purely on word counting and co-occurrence. They cannot understand meaning, context, or synonyms (e.g., "automobile" and "car" are treated as entirely separate entities).
- *Rigid Extractive Summaries:* The summarizer pulls verbatim sentences from the text, which can produce disjointed results when sentences rely heavily on surrounding context (e.g., pronouns like "He said...").
- *Fixed Topic Constraints:* LDA requires the user to specify the exact number of topics upfront. An incorrect number can force unrelated concepts together or unnecessarily fragment a single coherent topic.
- *Static Processing:* The system cannot iteratively learn or reason across multiple documents — it only processes the payload provided at execution time.

---

## How to Run (Milestone 1)

*1. Clone the Repository:*
bash
git clone https://github.com/Kushal425/GENai_SecA_P1.git
cd GENai_SecA_P1


*2. Create and Activate a Virtual Environment:*

macOS / Linux:
bash
python3 -m venv venv
source venv/bin/activate

Windows:
cmd
python -m venv venv
venv\Scripts\activate


*3. Install Dependencies:*
bash
pip install -r requirements.txt


*4. Download spaCy Language Model:*
bash
python -m spacy download en_core_web_sm


*5. Download NLTK Data:*
bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"


*6. Launch Streamlit App:*
bash
streamlit run app.py


The app will open automatically at *http://localhost:8501*

---

## Project Structure

ResearchScope/

│

├── data/   # Input documents and datasets

├── src/

│   ├── preprocessing.py          # Text cleaning, tokenization, lemmatization
│   ├── feature_extraction.py     # TF-IDF vectorization
│   ├── topic_model.py            # LDA topic modeling
│   ├── evaluation.py             # Coherence score evaluation
│   ├── keyword_extractor.py      # Keyword extraction logic
│   ├── summarizer.py             # Extractive summarization
│   ├── document_loader.py        # PDF and TXT file loading
│   └── visualizations.py         # Word cloud, bar charts
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── setup.sh                      # One-command environment setup script
├── README.md                     # Project documentation
├── architecture.png              # System architecture diagram
└── report.pdf                    # Mid-semester report


---
