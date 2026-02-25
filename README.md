# ResearchScope – Intelligent Topic Analyzer (Milestone 1)

A traditional NLP-based Streamlit web application that analyzes research text using Keyword Extraction, Topic Modeling (LDA), and Extractive Summarization.

---

## � Installation & Local Hosting

Follow these steps to set up the project locally on your machine.

### Prerequisites
Make sure you have **Python 3.10+** and **Git** installed on your system.

### Step 1. Clone the repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Kushal425/GENai_SecA_P1.git
cd GENai_SecA_P1
```

### Step 2. Create and activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3. Install Python Dependencies
Once the virtual environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

### Step 4. Download Required Language Models
The application relies on spaCy and NLTK models. Download them by running:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 5. Run the Application Locally
Start the Streamlit development server:
```bash
streamlit run app.py
```

### Step 6. Access the App
If your browser doesn't open automatically, navigate to the following URL in your web browser:
**➡️ http://localhost:8501**

---

*Note: To stop the local server at any time, press `Ctrl + C` in your terminal.*
