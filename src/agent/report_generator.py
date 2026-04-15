def format_report(state: dict) -> dict:
    """
    Takes the final agent state and returns a clean, validated report dict.
    Handles missing fields gracefully so the UI never crashes.
    """
    raw_report = state.get("report", {})
    query = state.get("query", "Research Report")

    title = raw_report.get("title") or query
    abstract = raw_report.get("abstract") or "No abstract was generated."
    key_findings = raw_report.get("key_findings") or []
    conclusion = raw_report.get("conclusion") or "No conclusion was generated."
    sources = raw_report.get("sources") or []

    # Ensure sources are properly formatted dicts
    clean_sources = []
    for s in sources:
        if isinstance(s, dict):
            clean_sources.append(
                {"title": s.get("title", "Untitled Source"), "url": s.get("url", "")}
            )

    return {
        "title": title,
        "abstract": abstract,
        "key_findings": key_findings,
        "conclusion": conclusion,
        "sources": clean_sources,
    }
