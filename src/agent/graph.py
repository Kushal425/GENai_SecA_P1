from langgraph.graph import StateGraph, END
from src.agent.state import ResearchState
from src.agent.nodes import (
    search_node,
    retrieve_node,
    validate_node,
    summarize_node,
    report_node,
)

def build_research_graph():
    """Build and compile the LangGraph research agent workflow."""
    graph = StateGraph(ResearchState)

    # Add all 5 nodes
    graph.add_node("search", search_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("report", report_node)

    # Define the linear edge flow
    graph.set_entry_point("search")
    graph.add_edge("search", "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", "summarize")
    graph.add_edge("summarize", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Compiled graph — imported by app.py
research_graph = build_research_graph()


def run_research_agent(query: str) -> dict:
    """
    Entry point: run the full research pipeline for a given query.
    Returns the final ResearchState dict.
    """
    initial_state = ResearchState(
        query=query,
        search_results=[],
        retrieved_texts=[],
        validated_sources=[],
        llm_summary="",
        report={},
        status="Starting...",
        error=None,
    )
    result = research_graph.invoke(initial_state)
    return result
