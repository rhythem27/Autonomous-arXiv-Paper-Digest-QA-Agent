"""SQLite checkpointer session management and thread persistence for LangGraph."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, Optional, Union

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.state import AgentState
from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def get_sqlite_saver(db_path: Optional[Union[str, Path]] = None) -> SqliteSaver:
    """Initialize a persistent SQLite-backed LangGraph checkpointer.

    Ensures checkpoint tables are configured and ready to accept graph states.

    Args:
        db_path: Path to the SQLite database file (defaults to settings.sqlite_db_path).
                 Pass ':memory:' for isolated ephemeral execution.

    Returns:
        SqliteSaver: Initialized SQLite checkpointer instance.
    """
    path_str = str(db_path or settings.sqlite_db_path)

    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Connecting SQLite checkpointer to: '{path_str}'")
    else:
        logger.debug("Connecting in-memory SQLite checkpointer")

    conn = sqlite3.connect(path_str, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@contextmanager
def get_sqlite_checkpointer(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[SqliteSaver, None, None]:
    """Context manager for temporary or transactional SQLite checkpointing.

    Args:
        db_path: Path to database or ':memory:'.

    Yields:
        SqliteSaver: Managed checkpointer instance.
    """
    path_str = str(db_path or settings.sqlite_db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(path_str) as saver:
        saver.setup()
        yield saver


def get_thread_config(session_id: str) -> Dict[str, Any]:
    """Format the LangGraph thread configuration dictionary for a given session.

    Args:
        session_id: Unique session or thread identifier string.

    Returns:
        Dict[str, Any]: Config dict required by LangGraph's `.invoke()` and `.stream()`.
    """
    return {"configurable": {"thread_id": session_id}}


def get_session_state(
    checkpointer: SqliteSaver,
    session_id: str,
) -> Optional[AgentState]:
    """Retrieve the latest persisted state snapshot for a given session ID.

    Args:
        checkpointer: Active SqliteSaver checkpointer instance.
        session_id: Target session thread identifier.

    Returns:
        Optional[AgentState]: Restored AgentState dictionary if found, else None.
    """
    config = get_thread_config(session_id)
    checkpoint_tuple = checkpointer.get_tuple(config)

    if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
        logger.debug(f"No existing checkpoint found for session: '{session_id}'")
        return None

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    return channel_values if channel_values else None
