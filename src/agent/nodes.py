"""Pure, modular LangGraph node implementations for Autonomous arXiv Paper Digest & QA Agent.

Implements all 7 assessment lifecycle stages along with diagnostic zero-results handling
and metadata fallback nodes.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.config import settings
from src.logger import get_logger
from src.agent.state import AgentState, PaperMetadata, ParsedPaper, ExecutiveBriefing
from src.tools.arxiv_client import (
    parse_query,
    get_arxiv_client,
    fetch_paper_by_id,
    search_papers_by_topic,
)
from src.parsers.pdf_parser import download_pdf, parse_pdf
from src.vectorstore.chunker import chunk_paper
from src.vectorstore.qdrant_store import (
    index_chunks,
    search_chunks,
    get_collection_name,
    get_embedding_model,
)
from src.llm.gemini_client import generate_briefing, answer_grounded_qa

logger = get_logger("agent_nodes")


def node_query_understanding(state: AgentState) -> Dict[str, Any]:
    """Node 1: Parses raw query into query_type, extracted_id, or search_keywords."""
    raw_query = state.get("raw_query", "")
    logger.info(f"Executing query_understanding node for input: {raw_query!r}")

    parsed = parse_query(raw_query)
    return {
        "query_type": parsed.query_type,
        "extracted_id": parsed.clean_id,
        "search_keywords": parsed.search_query,
        "status": "query_understood",
    }


def node_arxiv_retrieval(state: AgentState) -> Dict[str, Any]:
    """Node 2: Queries the official arXiv API Atom feed by ID or topic search."""
    query_type = state.get("query_type", "topic_search")
    extracted_id = state.get("extracted_id")
    search_keywords = state.get("search_keywords") or state.get("raw_query", "")

    client = get_arxiv_client()
    candidate_papers: List[PaperMetadata] = []
    selected_paper: Optional[PaperMetadata] = None

    try:
        if query_type == "arxiv_id" and extracted_id:
            logger.info(f"Fetching paper by direct arXiv ID: {extracted_id}")
            if hasattr(client, "fetch_by_id"):
                paper = client.fetch_by_id(extracted_id)
            else:
                paper = fetch_paper_by_id(extracted_id, client=client)
            if paper:
                candidate_papers = [paper]
                selected_paper = paper
        else:
            logger.info(f"Searching arXiv papers for topic: {search_keywords!r}")
            if hasattr(client, "search_papers"):
                candidate_papers = client.search_papers(
                    search_keywords,
                    max_results=settings.arxiv_max_results,
                )
            else:
                candidate_papers = search_papers_by_topic(
                    search_keywords,
                    max_results=settings.arxiv_max_results,
                    client=client,
                )

        if not candidate_papers:
            logger.warning(f"No papers retrieved for query: {search_keywords or extracted_id}")
            return {
                "candidate_papers": [],
                "selected_paper": None,
                "status": "zero_results",
            }

        return {
            "candidate_papers": candidate_papers,
            "selected_paper": selected_paper,
            "status": "retrieved",
        }

    except Exception as e:
        logger.error(f"Error during arXiv retrieval: {e}")
        return {
            "candidate_papers": [],
            "selected_paper": None,
            "error_message": f"arXiv retrieval error: {e}",
            "status": "retrieval_error",
        }


def _rank_candidates(query: str, candidates: List[PaperMetadata]) -> PaperMetadata:
    """Rank candidate papers using local semantic similarity against the user query."""
    if len(candidates) == 1:
        return candidates[0]

    try:
        model = get_embedding_model()
        texts = [f"{p.get('title', '')}. {p.get('abstract', '')}" for p in candidates]
        all_embeddings = list(model.embed([query] + texts))

        q_vec = np.array(all_embeddings[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        best_idx = 0
        best_score = -1.0
        for i, c_emb in enumerate(all_embeddings[1:]):
            c_vec = np.array(c_emb, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm
            score = float(np.dot(q_vec, c_vec))
            if score > best_score:
                best_score = score
                best_idx = i

        logger.info(f"Semantic ranking selected candidate #{best_idx + 1} with score {best_score:.4f}")
        return candidates[best_idx]
    except Exception as e:
        logger.warning(f"Candidate ranking fallback to first item: {e}")
        return candidates[0]


def node_selection_ranking(state: AgentState) -> Dict[str, Any]:
    """Node 3: Selects the targeted paper candidate from retrieved candidates."""
    selected_paper = state.get("selected_paper")
    if selected_paper:
        return {
            "selected_paper": selected_paper,
            "status": "paper_selected",
        }

    candidates = state.get("candidate_papers", [])
    if not candidates:
        return {
            "selected_paper": None,
            "status": "zero_results",
        }

    query = state.get("search_keywords") or state.get("raw_query", "")
    ranked_paper = _rank_candidates(query, candidates)

    return {
        "selected_paper": ranked_paper,
        "status": "paper_selected",
    }


def node_fetch_parse(state: AgentState) -> Dict[str, Any]:
    """Node 4: Streams PDF into cache and extracts structured sections with PyMuPDF."""
    paper = state.get("selected_paper")
    if not paper:
        return {
            "error_message": "No paper selected for PDF fetch and parse",
            "status": "parse_failed",
        }

    arxiv_id = paper.get("arxiv_id", "")
    pdf_url = paper.get("pdf_url", "")

    try:
        logger.info(f"Downloading PDF for arXiv ID {arxiv_id} from {pdf_url}")
        pdf_path = download_pdf(arxiv_id=arxiv_id, pdf_url=pdf_url)

        logger.info(f"Parsing structured sections from PDF: {pdf_path}")
        parsed = parse_pdf(
            pdf_path,
            fallback_title=paper.get("title", ""),
            fallback_abstract=paper.get("abstract", ""),
        )

        parsing_status = parsed.get("parsing_status", "success")
        return {
            "parsed_paper": parsed,
            "status": "parsed" if parsing_status != "metadata_only" else "metadata_only",
        }

    except Exception as e:
        logger.error(f"Failed to fetch or parse PDF for {arxiv_id}: {e}")
        return {
            "error_message": f"PDF parsing error: {e}",
            "status": "parse_failed",
        }


def node_metadata_fallback(state: AgentState) -> Dict[str, Any]:
    """Fallback Node: Constructs a minimal ParsedPaper using metadata when PDF parsing fails."""
    paper = state.get("selected_paper") or {}
    logger.info(f"Executing metadata fallback for paper: {paper.get('title', 'Unknown')}")

    fallback_parsed: ParsedPaper = {
        "title": paper.get("title", "Unknown Title"),
        "abstract": paper.get("abstract", ""),
        "sections": {
            "Abstract": paper.get("abstract", ""),
        },
        "references": [],
        "full_text": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}",
        "parsing_status": "metadata_only",
    }

    return {
        "parsed_paper": fallback_parsed,
        "status": "metadata_fallback",
    }


def node_chunk_embed(state: AgentState) -> Dict[str, Any]:
    """Node 5: Chunks paper sections and indexes dense embeddings into Qdrant local."""
    paper = state.get("selected_paper")
    parsed = state.get("parsed_paper")

    if not paper or not parsed:
        return {
            "status": "chunk_embed_skipped",
            "error_message": "Missing paper data for chunking and embedding",
        }

    arxiv_id = paper.get("arxiv_id", "")
    try:
        logger.info(f"Chunking paper sections for arXiv ID {arxiv_id}...")
        chunks = chunk_paper(parsed, arxiv_id=arxiv_id)

        logger.info(f"Indexing {len(chunks)} chunks into Qdrant local...")
        coll_name = index_chunks(arxiv_id, chunks)

        return {
            "qdrant_collection_name": coll_name,
            "status": "indexed",
        }
    except Exception as e:
        logger.error(f"Error indexing chunks into Qdrant for {arxiv_id}: {e}")
        return {
            "error_message": f"Embedding/indexing error: {e}",
            "status": "index_failed",
        }


def node_summarize(state: AgentState) -> Dict[str, Any]:
    """Node 6: Generates Executive Briefing with mandatory limitations via Gemini 2.5 Flash."""
    paper = state.get("selected_paper")
    parsed = state.get("parsed_paper")

    if not paper or not parsed:
        return {
            "status": "summarize_failed",
            "error_message": "Missing selected or parsed paper data for summarization",
        }

    try:
        logger.info(f"Generating Executive Briefing for: {paper.get('title', '')}")
        briefing = generate_briefing(paper, parsed)
        return {
            "briefing": briefing,
            "status": "summarized",
        }
    except Exception as e:
        logger.error(f"Error generating Executive Briefing: {e}")
        return {
            "error_message": f"Summarization error: {e}",
            "status": "summarize_failed",
        }


def node_qa_answer(state: AgentState) -> Dict[str, Any]:
    """Node 7: Retrieves relevant chunks from Qdrant and returns grounded answers with anti-hallucination."""
    qa_messages = state.get("qa_messages", [])

    # Extract last human message
    user_query = ""
    for msg in reversed(qa_messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") in ("human", "user"):
            user_query = str(msg.content)
            break

    if not user_query:
        return {"status": "qa_skipped"}

    paper = state.get("selected_paper")
    arxiv_id = paper.get("arxiv_id") if paper else state.get("extracted_id", "")

    context_chunks = []
    if arxiv_id:
        try:
            context_chunks = search_chunks(arxiv_id, user_query, top_k=settings.top_k_chunks)
        except Exception as e:
            logger.warning(f"Chunk retrieval failed for {arxiv_id}: {e}")

    # Synthesize answer with anti-hallucination guardrail
    answer_text = answer_grounded_qa(
        question=user_query,
        context_chunks=context_chunks,
        conversation_history=qa_messages,
    )

    return {
        "qa_messages": [AIMessage(content=answer_text)],
        "status": "qa_answered",
    }


def node_handle_zero_results(state: AgentState) -> Dict[str, Any]:
    """Diagnostic Node: Provides informative guidance when no arXiv papers match the query."""
    raw_query = state.get("raw_query", "")
    guidance = (
        f"No arXiv papers were found matching '{raw_query}'. "
        f"Please verify the arXiv ID (e.g., '1706.03762') or try searching with broader technical keywords."
    )
    logger.info(f"Executed zero_results node for query: {raw_query}")

    return {
        "error_message": guidance,
        "status": "zero_results",
    }
