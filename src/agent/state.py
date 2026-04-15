from typing import TypedDict, Optional

class ResearchState(TypedDict):
    """Shared state passed between all LangGraph nodes."""
    query: str                        # user's research question
    search_results: list              # raw DuckDuckGo results
    retrieved_texts: list             # fetched page content per source
    validated_sources: list           # filtered, quality-checked sources
    llm_summary: str                  # LLM-generated summary
    report: dict                      # final structured report
    status: str                       # current step (for UI status indicator)
    error: Optional[str]              # error message if any step fails
