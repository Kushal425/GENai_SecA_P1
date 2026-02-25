import PyPDF2
import io

def load_pdf(file) -> str:
    """Extract text from an uploaded PDF file object."""
    text = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception as e:
        return f"[Error reading PDF: {e}]"
    return "\n".join(text)

def load_txt(file) -> str:
    """Read a plain text file object."""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error reading file: {e}]"

def load_uploaded_files(uploaded_files) -> list[dict]:
    """
    Process a list of Streamlit UploadedFile objects.
    Returns a list of dicts: {"name": filename, "text": raw_text}
    """
    documents = []
    for uf in uploaded_files:
        if uf.name.endswith(".pdf"):
            text = load_pdf(uf)
        else:
            text = load_txt(uf)
        if text.strip():
            documents.append({"name": uf.name, "text": text})
    return documents
