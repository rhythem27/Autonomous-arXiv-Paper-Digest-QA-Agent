"""Stateful LangGraph orchestration graph for Autonomous arXiv Paper Digest & QA Agent.

Wires pipeline nodes with deterministic transitions, conditional routing branches,
interactive QA multi-turn dispatching, and SQLite checkpointer integration.
"""

from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.logger import get_logger
from src.agent.state import AgentState
from src.agent.nodes import (
    node_query_understanding,
    node_arxiv_retrieval,
    node_selection_ranking,
    node_fetch_parse,
    node_metadata_fallback,
    node_chunk_embed,
    node_summarize,
    node_qa_answer,
    node_handle_zero_results,
)
from src.agent.persistence import get_sqlite_saver

logger = get_logger("agent_graph")


def route_entrypoint(state: AgentState) -> str:
    """Determine whether to run the paper ingestion pipeline or dispatch to interactive QA answering.

    If the thread already contains a summarized paper and has a trailing unhandled user question
    in qa_messages, route directly to 'qa_answer'. Otherwise, start from 'query_understanding'.
    """
    selected_paper = state.get("selected_paper")
    briefing = state.get("briefing")
    qa_messages = state.get("qa_messages", [])

    if selected_paper and briefing and qa_messages:
        last_msg = qa_messages[-1]
        is_human = isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", "") in ("human", "user")
        if is_human:
            logger.info("Routing entrypoint directly to 'qa_answer' for follow-up question.")
            return "qa_answer"

    logger.info("Routing entrypoint to 'query_understanding' for paper digestion pipeline.")
    return "query_understanding"


def route_post_retrieval(state: AgentState) -> str:
    """Route after arXiv retrieval based on candidate count and query type.

    - 0 candidates -> 'handle_zero_results'
    - Direct arXiv ID -> 'fetch_parse' (bypasses ranking)
    - Topic search -> 'selection_ranking'
    """
    candidates = state.get("candidate_papers", [])
    if not candidates or state.get("status") in ("zero_results", "retrieval_error"):
        logger.warning("Zero papers retrieved; routing to 'handle_zero_results'.")
        return "handle_zero_results"

    if state.get("query_type") == "arxiv_id":
        logger.info("Direct arXiv ID query; routing directly to 'fetch_parse'.")
        return "fetch_parse"

    logger.info("Topic search query; routing to 'selection_ranking'.")
    return "selection_ranking"


def route_post_parse(state: AgentState) -> str:
    """Route after PDF parsing based on text extraction success.

    - Scanned, corrupt, or failed PDF -> 'metadata_fallback'
    - Successfully parsed structured sections -> 'chunk_embed'
    """
    parsed = state.get("parsed_paper")
    parsing_status = parsed.get("parsing_status", "") if parsed else ""
    state_status = state.get("status", "")

    if not parsed or parsing_status in ("metadata_only", "failed") or state_status in ("metadata_only", "parse_failed"):
        logger.warning("PDF parsing produced metadata only or failed; routing to 'metadata_fallback'.")
        return "metadata_fallback"

    logger.info("PDF parsed successfully; routing to 'chunk_embed'.")
    return "chunk_embed"


def build_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Construct and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional checkpoint saver (e.g., SqliteSaver). If None, defaults
                      to initializing a SqliteSaver from settings.sqlite_db_path.

    Returns:
        CompiledStateGraph: The compiled executable state graph.
    """
    logger.info("Constructing Autonomous arXiv Paper Digest & QA StateGraph...")

    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("query_understanding", node_query_understanding)
    workflow.add_node("arxiv_retrieval", node_arxiv_retrieval)
    workflow.add_node("selection_ranking", node_selection_ranking)
    workflow.add_node("fetch_parse", node_fetch_parse)
    workflow.add_node("metadata_fallback", node_metadata_fallback)
    workflow.add_node("chunk_embed", node_chunk_embed)
    workflow.add_node("summarize", node_summarize)
    workflow.add_node("qa_answer", node_qa_answer)
    workflow.add_node("handle_zero_results", node_handle_zero_results)

    # 2. Add Entrypoint Routing
    workflow.add_conditional_edges(
        START,
        route_entrypoint,
        {
            "query_understanding": "query_understanding",
            "qa_answer": "qa_answer",
        },
    )

    # 3. Add Deterministic Edge: query_understanding -> arxiv_retrieval
    workflow.add_edge("query_understanding", "arxiv_retrieval")

    # 4. Add Conditional Edge: arxiv_retrieval -> (zero_results | selection_ranking | fetch_parse)
    workflow.add_conditional_edges(
        "arxiv_retrieval",
        route_post_retrieval,
        {
            "handle_zero_results": "handle_zero_results",
            "selection_ranking": "selection_ranking",
            "fetch_parse": "fetch_parse",
        },
    )

    # 5. Add Deterministic Edge: selection_ranking -> fetch_parse
    workflow.add_edge("selection_ranking", "fetch_parse")

    # 6. Add Conditional Edge: fetch_parse -> (chunk_embed | metadata_fallback)
    workflow.add_conditional_edges(
        "fetch_parse",
        route_post_parse,
        {
            "chunk_embed": "chunk_embed",
            "metadata_fallback": "metadata_fallback",
        },
    )

    # 7. Add Ingestion Convergence: chunk_embed and metadata_fallback -> summarize
    workflow.add_edge("chunk_embed", "summarize")
    workflow.add_edge("metadata_fallback", "summarize")

    # 8. Terminal Edges -> END
    workflow.add_edge("summarize", END)
    workflow.add_edge("qa_answer", END)
    workflow.add_edge("handle_zero_results", END)

    # 9. Compile with checkpointer
    if checkpointer is None:
        try:
            checkpointer = get_sqlite_saver()
        except Exception as e:
            logger.warning(f"Could not initialize default SQLite saver: {e}; compiling without checkpointer.")
            checkpointer = None

    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("StateGraph successfully compiled.")
    return compiled_graph
