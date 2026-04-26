# 🔬 ResearchScope: Intelligent Research Analyzer

> An autonomous, AI-driven educational and research platform that leverages classical machine learning for deep document analysis and a LangGraph-based agentic workflow for live, open-web research synthesis.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Classical_NLP-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

**🌐 Live Demo:** [https://researchscopegenai.streamlit.app/](https://researchscopegenai.streamlit.app/)

## 📖 Overview

ResearchScope is a dual-mode full-stack AI application designed to support students, researchers, and knowledge workers through data-driven insights and autonomous web synthesis. Built as the Capstone Project for the Generative AI course, it combines **two core milestones** into a single, cohesive platform switchable via the UI:

1. **Document Analysis Engine (Classical NLP - Milestone 1):** Analyzes uploaded local documents (PDFs/TXTs) using Scikit-Learn and Gensim to cluster latent topics (LDA), extract high-value vocabulary (TF-IDF), and generate purely extractive, verifiable summaries without any LLM hallucination risk.
2. **Agentic AI Research Assistant (LLM + Web RAG - Milestone 2):** An autonomous AI agent built with **LangGraph**. It ingests open-ended research queries and dynamically fetches live content via DuckDuckGo and BeautifulSoup. It acts as an autonomous researcher, navigating the live web, validating sources, and utilizing a free-tier Groq LLM to synthesize factual, structured research reports dynamically.

## ✨ Key Features

- **Autonomous Agent Workflow:** A 5-node reasoning workflow orchestrated by LangGraph (`Search` $\rightarrow$ `Retrieve` $\rightarrow$ `Validate` $\rightarrow$ `Summarize` $\rightarrow$ `Report`), enabling the AI to act as a stateful, goal-oriented researcher.
- **Live Web RAG (Retrieval-Augmented Generation):** The agent autonomously fetches multi-source web content and injects it into the LLM context, effectively bypassing the need for static vector databases.
- **Predictive & Classical Analytics:** Interactive data visualizations (Matplotlib/WordClouds) showing topic coherence ($C_v$) and TF-IDF term weights for local documents.
- **Strict Anti-Hallucination Measures:** The system injects retrieved HTML directly into the agent's system prompt with strict bounding instructions, ensuring the AI maintains an academic, evidence-only tone.
- **Dynamic PDF Export:** Built-in integration with ReportLab to dynamically style and generate A4 PDF research reports containing the LLM output and source citations.

## 🛠️ Technology Stack

- **Framework:** Streamlit (Frontend & Dashboard)
- **Agentic Orchestration:** LangGraph, LangChain Core
- **Classical NLP / ML:** Scikit-Learn (TF-IDF), Gensim (LDA Topic Modeling), NLTK, spaCy
- **Live Web Retrieval:** DuckDuckGo Search (`ddgs`), HTTPX, BeautifulSoup4
- **Large Language Model:** Groq API (`llama-3.3-70b-versatile`) for ultra-fast, free-tier inference
- **PDF Generation:** ReportLab

## 👨‍💻 Team & Contributions

This platform was developed collaboratively as a group project. 

- **Kushal Sarkar (Project Lead):** Architected the core LangGraph state machine, implemented the agentic workflow nodes (Search, Retrieve, Validate, Summarize, Report), and integrated the Groq LLM API.
- **Chinmay Soni:** Developed the PDF Export extension using ReportLab, ensuring the dynamic generation of styled, downloadable academic reports.
- **Lakshya Bapna:** Engineered the Streamlit frontend UI for Milestone 2, including the live-streaming LangGraph status indicators and result tabs.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Kushal425/GENai_SecA_P1.git
cd GENai_SecA_P1
```

### 2. Create a Virtual Environment (Recommended)
It's best practice to use a virtual environment to manage dependencies.
*macOS / Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```
*Windows:*
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Set Environment Variables
Create a file at `.streamlit/secrets.toml` and add your Groq API key (get one for free at [groq.com](https://console.groq.com/)).
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
*(Note: Milestone 1 works completely offline without an API key).*

### 5. Run the Application
```bash
streamlit run app.py
```

## 🧠 How It Works (Architecture)

1. **User interacts with the Dashboard** to either upload documents (M1) or enter a live research query (M2).
2. **`src/topic_model.py` & `src/keyword_extractor.py`** processes offline documents, generating TF-IDF matrices and LDA topic clusters to surface latent themes.
3. **`src/agent/graph.py`** initializes a LangGraph `StateGraph` for live queries. The user's query is injected into the `ResearchState`.
4. **The Agent navigates the web**. The pipeline autonomously triggers DuckDuckGo, scrapes raw HTML from the top URLs, filters out low-quality sites, and compiles the context.
5. **`src/agent/nodes.py`** passes the validated context to the `llama-3.3-70b-versatile` model, synthesizing a highly structured Title, Abstract, Key Findings, and Conclusion.

## 🌐 Deployment

**The live application is deployed here:** [https://researchscopegenai.streamlit.app/](https://researchscopegenai.streamlit.app/)

This project is built to be seamlessly deployed to platforms like **Streamlit Community Cloud**.

### Streamlit Community Cloud Deployment
1. Push your repository to GitHub (ensure your `.streamlit/secrets.toml` is in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Deploy the repository by selecting `app.py` as the main file.
4. In the Streamlit dashboard, go to **Advanced Settings -> Secrets** and add your API key: `GROQ_API_KEY="your_key"`.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Kushal425/GENai_SecA_P1/issues).

## 📝 License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.