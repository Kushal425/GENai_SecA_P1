import os
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_groq import ChatGroq

# ── LLM Setup ──────────────────────────────────────────────────────────────────
def get_llm():
    """Load Groq LLM. Reads API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.3)


# ── Node 1: Web Search ─────────────────────────────────────────────────────────
def search_node(state: dict) -> dict:
    """Search the web using DuckDuckGo. No API key required."""
    state["status"] = "Searching the web..."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(state["query"], max_results=6))
        state["search_results"] = results
    except Exception as e:
        state["search_results"] = []
        state["error"] = f"Search failed: {str(e)}"
    return state


# ── Node 2: Retrieve Page Content ─────────────────────────────────────────────
def retrieve_node(state: dict) -> dict:
    """Fetch and parse full text from each search result URL."""
    state["status"] = "Retrieving source content..."
    texts = []
    for result in state["search_results"]:
        url = result.get("href", "")
        title = result.get("title", "")
        snippet = result.get("body", "")
        try:
            resp = httpx.get(url, timeout=8, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            full_text = " ".join(p.get_text() for p in paragraphs)[:3000]
            text = full_text if len(full_text) > 200 else snippet
        except Exception:
            # Fallback to DuckDuckGo snippet if fetch fails
            text = snippet
        texts.append({"url": url, "title": title, "text": text})
    state["retrieved_texts"] = texts
    return state


# ── Node 3: Validate Sources ───────────────────────────────────────────────────
def validate_node(state: dict) -> dict:
    """Filter out empty, duplicate, or low-quality sources."""
    state["status"] = "Validating sources..."
    valid = []
    seen_urls = set()
    for item in state["retrieved_texts"]:
        url = item.get("url", "")
        text = item.get("text", "")
        is_unique = url not in seen_urls
        has_content = len(text.strip()) > 100
        if is_unique and has_content:
            valid.append(item)
            seen_urls.add(url)
    state["validated_sources"] = valid[:5]  # Keep top 5 sources
    return state


# ── Node 4: Summarize with LLM ─────────────────────────────────────────────────
def summarize_node(state: dict) -> dict:
    """Send validated source content to Groq LLM for summarization."""
    state["status"] = "Summarizing with LLM..."
    try:
        llm = get_llm()
        combined = "\n\n".join(
            f"Source: {s['title']}\n{s['text'][:1200]}"
            for s in state["validated_sources"]
        )
        prompt = f"""You are a research assistant. Based on the following web sources, write a comprehensive summary about:

"{state['query']}"

Sources:
{combined}

Instructions:
- Write 3 to 5 clear, factual paragraphs
- Only use information found in the sources above
- Do not hallucinate or add outside information
- Be concise and academic in tone"""

        response = llm.invoke(prompt)
        state["llm_summary"] = response.content
    except Exception as e:
        state["llm_summary"] = (
            "LLM summarization failed. Please check your GROQ_API_KEY."
        )
        state["error"] = f"LLM error: {str(e)}"
    return state


# ── Node 5: Generate Structured Report ────────────────────────────────────────
def report_node(state: dict) -> dict:
    """Use LLM to generate a structured research report from the summary."""
    state["status"] = "Generating structured report..."
    try:
        llm = get_llm()
        prompt = f"""Based on this research summary about "{state['query']}", generate a structured report.

Summary:
{state['llm_summary']}

Return the report in EXACTLY this format with these exact labels:
TITLE: [A descriptive research title]
ABSTRACT: [2-3 sentence overview of the topic and findings]
KEY FINDINGS:
- [Finding 1]
- [Finding 2]
- [Finding 3]
- [Finding 4]
- [Finding 5]
CONCLUSION: [2-3 sentence conclusion and implications]"""

        response = llm.invoke(prompt)
        state["report"] = _parse_report(
            response.content, state["query"], state["validated_sources"]
        )
    except Exception as e:
        state["report"] = {
            "title": state["query"],
            "abstract": state.get("llm_summary", ""),
            "key_findings": ["Report generation failed — see raw summary tab."],
            "conclusion": "An error occurred during report generation.",
            "sources": [
                {"title": s.get("title", "Source"), "url": s.get("url", "")}
                for s in state["validated_sources"]
            ],
        }
        state["error"] = f"Report node error: {str(e)}"

    state["status"] = "Complete"
    return state


# ── Internal: Parse LLM Report Output ────────────────────────────────────────
def _parse_report(raw: str, query: str, sources: list) -> dict:
    """Parse the LLM's formatted text response into a structured dict."""
    report = {
        "title": query,
        "abstract": "",
        "key_findings": [],
        "conclusion": "",
        "sources": [
            {"title": s.get("title", "Source"), "url": s.get("url", "")}
            for s in sources
        ],
    }
    current_section = None

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("TITLE:"):
            report["title"] = line.replace("TITLE:", "").strip()
        elif line.startswith("ABSTRACT:"):
            report["abstract"] = line.replace("ABSTRACT:", "").strip()
            current_section = "abstract"
        elif line.startswith("KEY FINDINGS:"):
            current_section = "findings"
        elif line.startswith("CONCLUSION:"):
            report["conclusion"] = line.replace("CONCLUSION:", "").strip()
            current_section = "conclusion"
        elif current_section == "findings" and line.startswith("-"):
            report["key_findings"].append(line.lstrip("- ").strip())
        elif current_section == "abstract" and line and not any(
            line.startswith(k) for k in ["KEY FINDINGS", "CONCLUSION", "TITLE"]
        ):
            report["abstract"] += (" " + line) if report["abstract"] else line
        elif current_section == "conclusion" and line and not any(
            line.startswith(k) for k in ["KEY FINDINGS", "ABSTRACT", "TITLE"]
        ):
            report["conclusion"] += (" " + line) if report["conclusion"] else line

    return report
