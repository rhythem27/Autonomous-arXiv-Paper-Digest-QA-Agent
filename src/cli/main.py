"""Rich interactive CLI application for Autonomous arXiv Paper Digest & QA Agent.

Provides animated execution spinners, presentation-ready briefing panels,
and a multi-turn grounded QA conversational loop.
"""

import logging
import sys
import uuid
import warnings
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt
from langchain_core.messages import HumanMessage, AIMessage

# Ensure UTF-8 output encoding across Windows consoles to support math symbols and Unicode
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress noisy third-party warnings (e.g. langchain fixed sampling warnings)
warnings.filterwarnings("ignore")

from src.logger import get_logger, set_cli_silent
from src.agent.state import create_initial_state
from src.agent.graph import build_agent_graph
from src.agent.persistence import get_thread_config, get_sqlite_saver


def _silence_console_loggers():
    """Silence noisy module logs in interactive CLI mode so Rich UI remains pristine."""
    set_cli_silent(True)
    warnings.filterwarnings("ignore")
    try:
        import google.genai.models
        google.genai.models.Models._logged_afc_warning = True
        google.genai.models.AsyncModels._logged_afc_warning = True
    except Exception:
        pass
    for name in [
        "agent_graph",
        "agent_nodes",
        "gemini_client",
        "src.tools.arxiv_client",
        "src.parsers.pdf_parser",
        "src.vectorstore.chunker",
        "src.vectorstore.qdrant_store",
        "src.agent.persistence",
        "cli",
        "arxiv_agent",
        "google_genai",
        "google_genai.models",
        "google",
    ]:
        l = logging.getLogger(name)
        l.setLevel(logging.ERROR)
        for h in l.handlers:
            h.setLevel(logging.ERROR)

logger = get_logger("cli")
console = Console(legacy_windows=False)


app = typer.Typer(
    name="arxiv-agent",
    help="Autonomous arXiv Paper Digest & Grounded QA Agent",
    add_completion=False,
)

STAGE_LABELS = {
    "query_understanding": "[1/6] Classifying query intent & identifier...",
    "arxiv_retrieval": "[2/6] Querying official arXiv API Atom feed...",
    "selection_ranking": "[3/6] Ranking candidates with semantic embeddings...",
    "fetch_parse": "[4/6] Streaming PDF & extracting sections with PyMuPDF...",
    "metadata_fallback": "[!] PDF parsing fallback: using abstract metadata...",
    "chunk_embed": "[5/6] Section-aware chunking & indexing into Qdrant Local...",
    "summarize": "[6/6] Generating Executive Briefing with Gemini 2.5 Flash...",
    "handle_zero_results": "[!] No candidate papers found...",
    "qa_answer": "Retrieving relevant sections & synthesizing grounded answer...",
}


def print_banner():
    """Render the application banner with styling."""
    banner_text = (
        "[bold cyan]Autonomous arXiv Paper Digest & QA Agent[/bold cyan]\n"
        "[dim]Powered by LangGraph • Gemini 2.5 Flash • Qdrant Local • BGE-small • PyMuPDF • SQLite[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", expand=False))


def display_candidates_table(candidates: list):
    """Render a table displaying retrieved candidate papers."""
    if not candidates or len(candidates) <= 1:
        return

    table = Table(title="Retrieved arXiv Candidate Papers", border_style="blue", show_lines=True)
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("Title", style="bold white")
    table.add_column("Primary Category", style="green", width=18)
    table.add_column("Published", style="magenta", width=12)

    for i, p in enumerate(candidates, 1):
        table.add_row(
            str(i),
            p.get("arxiv_id", "N/A"),
            p.get("title", "N/A"),
            p.get("primary_category", "N/A"),
            p.get("published", "N/A")[:10],
        )

    console.print(table)


def run_ingestion_pipeline(graph, query: str, session_id: str) -> dict:
    """Execute the paper digestion pipeline with animated node progress."""
    initial_state = create_initial_state(raw_query=query, session_id=session_id)
    config = get_thread_config(session_id)

    final_state = initial_state
    with console.status("[bold green]Starting ingestion pipeline...", spinner="dots") as status:
        for output in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_update in output.items():
                label = STAGE_LABELS.get(node_name, f"Executing {node_name}...")
                status.update(f"[bold cyan]{label}[/bold cyan]")
                logger.info(f"CLI observed node completion: {node_name}")
                # Merge updates
                final_state = {**final_state, **node_update}

    return final_state


def run_qa_turn(graph, question: str, session_id: str) -> str:
    """Execute a single QA turn on an existing persisted session."""
    config = get_thread_config(session_id)
    input_update = {
        "qa_messages": [HumanMessage(content=question)]
    }

    with console.status("[bold green]Synthesizing grounded answer...", spinner="dots") as status:
        status.update("[bold cyan]Retrieving relevant sections & citing sources...[/bold cyan]")
        output_state = graph.invoke(input_update, config=config)

    qa_messages = output_state.get("qa_messages", [])
    if qa_messages:
        last_msg = qa_messages[-1]
        if isinstance(last_msg, AIMessage):
            return str(last_msg.content)
        return getattr(last_msg, "content", "No answer generated.")
    return "No answer generated."


def interactive_qa_loop(graph, session_id: str, paper_title: str):
    """Run an interactive conversation loop for follow-up questions."""
    console.print(
        Panel(
            f"[bold green]Interactive QA Session Active[/bold green]\n"
            f"[dim]Paper: {paper_title}[/dim]\n"
            f"[yellow]Type 'exit' to quit, 'status' for thread info, 'help' for commands[/yellow]",
            border_style="green",
            expand=False,
        )
    )

    while True:
        try:
            prompt_str = f"[bold cyan]Ask a question[/bold cyan]"
            question = Prompt.ask(prompt_str).strip()

            if not question:
                continue

            lowered = question.lower()
            if lowered in ("exit", "quit", "/exit", "/quit", "q"):
                console.print("\n[bold green]Thank you for using arXiv Digest Agent. Goodbye![/bold green]")
                break

            if lowered in ("help", "/help", "?"):
                help_table = Table(title="Available Commands", border_style="yellow")
                help_table.add_column("Command", style="bold cyan")
                help_table.add_column("Action", style="white")
                help_table.add_row("exit / quit", "Exit the interactive session")
                help_table.add_row("status", "Show current session and paper details")
                help_table.add_row("help", "Show this command guide")
                help_table.add_row("<any question>", "Ask a research question grounded in the paper")
                console.print(help_table)
                continue

            if lowered in ("status", "/status"):
                status_table = Table(title="Session Status", border_style="magenta")
                status_table.add_column("Property", style="bold")
                status_table.add_column("Value", style="cyan")
                status_table.add_row("Session ID", session_id)
                status_table.add_row("Paper", paper_title)
                console.print(status_table)
                continue

            # Answer grounded question
            answer = run_qa_turn(graph, question, session_id)
            console.print(
                Panel(
                    Markdown(answer),
                    title=f"[bold green]Grounded Answer[/bold green]",
                    border_style="green",
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session interrupted. Exiting...[/bold yellow]")
            break


@app.command()
def cli_main(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="arXiv ID (e.g. '1706.03762'), arXiv URL, or natural language research topic",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Optional session thread ID for SQLite state checkpointing",
    ),
    question: Optional[str] = typer.Option(
        None,
        "--question",
        help="Optional direct question to answer immediately about the paper",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Whether to enter interactive QA mode after generating the briefing",
    ),
):
    """Main CLI entrypoint for Autonomous arXiv Paper Digest & QA Agent."""
    _silence_console_loggers()
    print_banner()

    # Resolve query
    if not query:
        query = Prompt.ask(
            "\n[bold yellow]Enter arXiv ID, paper URL, or research topic[/bold yellow]"
        ).strip()

    if not query:
        console.print("[bold red]No query provided. Exiting.[/bold red]")
        raise typer.Exit(code=1)

    # Initialize session and graph
    active_session_id = session_id or f"session_{uuid.uuid4().hex[:10]}"
    console.print(f"[dim]Thread ID: {active_session_id}[/dim]\n")

    checkpointer = get_sqlite_saver()
    graph = build_agent_graph(checkpointer=checkpointer)

    # Run ingestion pipeline
    state = run_ingestion_pipeline(graph, query, active_session_id)

    # Handle zero results
    if state.get("status") == "zero_results":
        error_msg = state.get("error_message", "No papers found matching your query.")
        console.print(
            Panel(
                f"[bold red]Zero Results[/bold red]\n{error_msg}",
                border_style="red",
            )
        )
        raise typer.Exit(code=0)

    # Display candidate table if topic search
    candidates = state.get("candidate_papers", [])
    if candidates and len(candidates) > 1:
        display_candidates_table(candidates)

    selected_paper = state.get("selected_paper")
    paper_title = selected_paper.get("title", "Research Paper") if selected_paper else "Research Paper"

    # Display Executive Briefing
    briefing = state.get("briefing")
    if briefing and briefing.get("markdown_output"):
        console.print(
            Panel(
                Markdown(briefing["markdown_output"]),
                title=f"[bold cyan]Executive Briefing: {paper_title}[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
    else:
        console.print(f"[bold yellow]Ingestion completed with status: {state.get('status')}[/bold yellow]")

    # Answer direct question if provided
    if question:
        console.print(f"\n[bold cyan]Processing Question:[/bold cyan] {question}")
        direct_answer = run_qa_turn(graph, question, active_session_id)
        console.print(
            Panel(
                Markdown(direct_answer),
                title="[bold green]Grounded Answer[/bold green]",
                border_style="green",
            )
        )

    # Enter interactive QA loop if requested
    if interactive and not question:
        interactive_qa_loop(graph, active_session_id, paper_title)


if __name__ == "__main__":
    app()
