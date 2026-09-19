"""Unit and integration tests for AgentState schema, message reduction, and SQLite persistence."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
import pytest

from src.agent.persistence import (
    get_session_state,
    get_sqlite_checkpointer,
    get_sqlite_saver,
    get_thread_config,
)
from src.agent.state import (
    AgentState,
    ExecutiveBriefing,
    create_initial_state,
)


class TestAgentStateSchema:
    """Unit tests for state dataclass/TypedDict validation and initializers."""

    def test_create_initial_state_defaults(self):
        """Verify default state creation and field populating."""
        state = create_initial_state("  KV-cache compression for LLMs  ")
        assert state["raw_query"] == "KV-cache compression for LLMs"
        assert state["session_id"].startswith("session_")
        assert state["query_type"] == "topic_search"
        assert state["candidate_papers"] == []
        assert state["selected_paper"] is None
        assert state["parsed_paper"] is None
        assert state["qdrant_collection_name"] is None
        assert state["briefing"] is None
        assert state["qa_messages"] == []
        assert state["status"] == "initialized"
        assert state["error_message"] is None

    def test_create_initial_state_custom_session(self):
        """Verify custom session ID injection."""
        custom_id = "test_custom_session_12345"
        state = create_initial_state("test query", session_id=custom_id)
        assert state["session_id"] == custom_id

    def test_executive_briefing_structure(self):
        """Verify ExecutiveBriefing typing and mandatory fields."""
        briefing: ExecutiveBriefing = {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "arxiv_id": "1706.03762",
            "publish_date": "2017-06-12",
            "link": "https://arxiv.org/abs/1706.03762",
            "why_it_matters": "Introduced the Transformer architecture which powers modern LLMs.",
            "problem_statement": "Recurrent networks cannot be parallelized across sequence lengths.",
            "method_approach": ["Multi-head self-attention", "Positional encoding"],
            "key_results": ["BLEU 28.4 on WMT English-to-German"],
            "limitations": ["Quadratic computational complexity with sequence length"],
            "suggested_follow_ups": ["Can attention be made linear in time and memory?"],
            "markdown_output": "# Attention Is All You Need\n\nSummary...",
        }
        assert briefing["arxiv_id"] == "1706.03762"
        assert len(briefing["limitations"]) == 1


class TestPersistenceUnit:
    """Unit tests for SQLite checkpointer initialization and thread configs."""

    def test_thread_config_structure(self):
        """Verify LangGraph thread configuration dictionary."""
        config = get_thread_config("sess_abc123")
        assert config == {"configurable": {"thread_id": "sess_abc123"}}

    def test_in_memory_sqlite_saver(self):
        """Verify in-memory checkpointer setup."""
        saver = get_sqlite_saver(":memory:")
        assert isinstance(saver, SqliteSaver)

    def test_context_manager_checkpointer(self):
        """Verify context manager creation of SqliteSaver."""
        with get_sqlite_checkpointer(":memory:") as saver:
            assert isinstance(saver, SqliteSaver)


class TestStatePersistenceAndRecovery:
    """Integration test compiling a LangGraph workflow with SQLite checkpointing."""

    def test_graph_state_persistence_and_message_appending(self):
        """Compile a simple 2-node graph with SQLite checkpointing and verify multi-turn state."""
        # 1. Define sample nodes updating state
        def node_one(state: AgentState) -> dict:
            return {
                "status": "node_one_complete",
                "qa_messages": [HumanMessage(content="What is the core idea?")],
            }

        def node_two(state: AgentState) -> dict:
            return {
                "status": "node_two_complete",
                "qa_messages": [AIMessage(content="The core idea is self-attention.")],
            }

        # 2. Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("step_one", node_one)
        workflow.add_node("step_two", node_two)
        workflow.add_edge(START, "step_one")
        workflow.add_edge("step_one", "step_two")
        workflow.add_edge("step_two", END)

        saver = get_sqlite_saver(":memory:")
        app = workflow.compile(checkpointer=saver)

        # 3. Execute graph on session_1
        session_id = "session_test_run"
        init_state = create_initial_state("attention query", session_id=session_id)
        config = get_thread_config(session_id)

        result = app.invoke(init_state, config=config)

        assert result["status"] == "node_two_complete"
        assert len(result["qa_messages"]) == 2
        assert result["qa_messages"][0].content == "What is the core idea?"
        assert result["qa_messages"][1].content == "The core idea is self-attention."

        # 4. Verify state recovery from SQLite checkpointer
        saved_state = get_session_state(saver, session_id)
        assert saved_state is not None
        assert saved_state["status"] == "node_two_complete"
        assert len(saved_state["qa_messages"]) == 2

        # 5. Execute turn 2 on the same session with follow-up input
        result_turn_2 = app.invoke(
            {"qa_messages": [HumanMessage(content="What are the limitations?")]},
            config=config,
        )

        # Message reducer (add_messages) must accumulate all messages across turns!
        assert len(result_turn_2["qa_messages"]) >= 3
        assert any("What are the limitations?" in m.content for m in result_turn_2["qa_messages"])
