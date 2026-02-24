# ResearchScope – Intelligent Topic Analyzer

A Streamlit web app that analyzes research text using **NLP techniques** — keyword extraction, LDA topic modeling, text summarization, and visualizations.

## Features
- 🔍 **TF-IDF Keyword Extraction**
- 📚 **LDA Topic Modeling** with coherence scoring
- 📝 **Automatic Text Summarization**
- ☁️ **Word Cloud & Keyword Bar Chart Visualizations**

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd GEN_AI
```

### 2. Run the setup script (recommended)
```bash
bash setup.sh
```

### 2. Or set up manually
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Required: download spaCy language model
python -m spacy download en_core_web_sm

# Required: download NLTK corpora
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Run the app
```bash
source venv/bin/activate
streamlit run app.py
```

## Project Structure
```
GEN_AI/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── setup.sh                # One-command environment setup
├── data/                   # Input data directory
└── src/
    ├── preprocessing.py    # Text cleaning, tokenization, lemmatization
    ├── feature_extraction.py  # TF-IDF vectorization
    ├── topic_model.py      # LDA topic modeling
    ├── evaluation.py       # Coherence score evaluation
    ├── keyword_extractor.py   # Keyword extraction
    ├── summarizer.py       # Text summarization
    └── visualizations.py  # Word cloud & charts
```

## Dependencies
See `requirements.txt`. Key libraries: `streamlit`, `spacy`, `nltk`, `scikit-learn`, `gensim`, `wordcloud`, `matplotlib`.
