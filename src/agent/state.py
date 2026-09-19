"""Agent state schema definitions for the LangGraph paper digest and QA pipeline."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import uuid

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.parsers.pdf_parser import ParsedPaper
from src.tools.arxiv_client import PaperMetadata


class ExecutiveBriefing(TypedDict):
    """Structured executive summary of a research paper produced by Gemini."""

    title: str
    authors: List[str]
    arxiv_id: str
    publish_date: str
    link: str
    why_it_matters: str  # 1-paragraph plain-English summary of significance
    problem_statement: str  # Concise problem definition the paper tackles
    method_approach: List[str]  # Bullet points detailing architecture/method
    key_results: List[str]  # Empirical findings, benchmarks, and claims
    limitations: List[str]  # Mandatory section: constraints, assumptions, weaknesses
    suggested_follow_ups: List[str]  # Actionable questions/extensions for researchers
    markdown_output: str  # Complete, beautiful formatted markdown rendering


class AgentState(TypedDict):
    """Shared typed state flowing through all LangGraph agent graph nodes."""

    session_id: str  # SQLite persistence session/thread identifier
    raw_query: str  # Original query or topic submitted by user
    query_type: str  # "arxiv_id", "topic_search", or "url"
    extracted_id: Optional[str]  # Canonical parsed arXiv ID if lookup
    search_keywords: Optional[str]  # Normalized keywords for topic search
    candidate_papers: List[PaperMetadata]  # Candidate paper records from arXiv API
    selected_paper: Optional[PaperMetadata]  # Targeted paper for digest & QA
    parsed_paper: Optional[ParsedPaper]  # Structured textual sections from PyMuPDF
    qdrant_collection_name: Optional[str]  # Collection name where chunks are indexed
    briefing: Optional[ExecutiveBriefing]  # Structured Executive Briefing artifact
    qa_messages: Annotated[List[BaseMessage], add_messages]  # Conversational QA history with message reducer
    error_message: Optional[str]  # Diagnostic message if errors or zero results occur
    status: str  # Pipeline stage status flag


def create_initial_state(
    raw_query: str,
    session_id: Optional[str] = None,
) -> AgentState:
    """Initialize a new default AgentState dictionary for graph execution.

    Args:
        raw_query: Raw user prompt, topic, or arXiv identifier.
        session_id: Optional session ID string. Generates UUID4 hex if omitted.

    Returns:
        AgentState: Fully typed and initialized state dictionary.
    """
    sid = session_id or f"session_{uuid.uuid4().hex[:10]}"
    return AgentState(
        session_id=sid,
        raw_query=raw_query.strip() if raw_query else "",
        query_type="topic_search",
        extracted_id=None,
        search_keywords=None,
        candidate_papers=[],
        selected_paper=None,
        parsed_paper=None,
        qdrant_collection_name=None,
        briefing=None,
        qa_messages=[],
        error_message=None,
        status="initialized",
    )
