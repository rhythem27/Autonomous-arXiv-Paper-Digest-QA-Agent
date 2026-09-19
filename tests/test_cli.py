"""Unit and integration tests for the Rich Interactive CLI interface.

Verifies:
- Command-line argument parsing and help menus
- Non-interactive batch execution with --query and --question
- Animated pipeline execution and briefing presentation
- Zero-results diagnostic panel rendering
- Interactive QA conversation loop and slash commands ('exit', 'help', 'status')
"""

from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from src.cli.main import app
from src.agent.state import PaperMetadata, ExecutiveBriefing

runner = CliRunner()

SAMPLE_BRIEFING: ExecutiveBriefing = {
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "arxiv_id": "1706.03762",
    "publish_date": "2017-06-12",
    "link": "https://arxiv.org/pdf/1706.03762.pdf",
    "why_it_matters": "Replaced recurrent models with multi-head self-attention.",
    "problem_statement": "Sequential training bottleneck in recurrent architectures.",
    "method_approach": ["Multi-head attention", "Positional encoding"],
    "key_results": ["28.4 BLEU on English-to-German"],
    "limitations": ["Quadratic memory complexity with sequence length"],
    "suggested_follow_ups": ["Can linear attention achieve similar results?"],
    "markdown_output": "# Executive Briefing: Attention Is All You Need\n\n## 1. Why This Paper Matters\nReplaced recurrent models.",
}

SAMPLE_PAPER: PaperMetadata = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani", "Noam Shazeer"],
    "published": "2017-06-12",
    "updated": "2017-06-12",
    "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
    "primary_category": "cs.CL",
    "categories": ["cs.CL"],
    "entry_id": "http://arxiv.org/abs/1706.03762v1",
}


class TestCliInterface:
    """Test Typer CLI options, flags, and execution modes."""

    def test_cli_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Autonomous arXiv Paper Digest & QA Agent" in result.stdout
        assert "--query" in result.stdout
        assert "--session-id" in result.stdout
        assert "--interactive" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_direct_query_no_interactive(self, mock_pipeline, mock_graph):
        mock_pipeline.return_value = {
            "status": "summarized",
            "selected_paper": SAMPLE_PAPER,
            "briefing": SAMPLE_BRIEFING,
            "candidate_papers": [SAMPLE_PAPER],
        }

        result = runner.invoke(app, ["--query", "1706.03762", "--no-interactive"])
        assert result.exit_code == 0
        assert "Autonomous arXiv Paper Digest & QA Agent" in result.stdout
        assert "Executive Briefing" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_qa_turn")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_direct_query_with_question(self, mock_pipeline, mock_qa, mock_graph):
        mock_pipeline.return_value = {
            "status": "summarized",
            "selected_paper": SAMPLE_PAPER,
            "briefing": SAMPLE_BRIEFING,
            "candidate_papers": [SAMPLE_PAPER],
        }
        mock_qa.return_value = "Based on [Section: Results]: The model achieves 28.4 BLEU."

        result = runner.invoke(
            app,
            [
                "--query",
                "1706.03762",
                "--question",
                "What BLEU score was achieved?",
                "--no-interactive",
            ],
        )

        assert result.exit_code == 0
        assert "Processing Question" in result.stdout
        assert "28.4 BLEU" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_zero_results_warning_panel(self, mock_pipeline, mock_graph):
        mock_pipeline.return_value = {
            "status": "zero_results",
            "error_message": "No arXiv papers found matching 'asdfghjkl123'.",
        }

        result = runner.invoke(app, ["--query", "asdfghjkl123", "--no-interactive"])
        assert result.exit_code == 0
        assert "Zero Results" in result.stdout
        assert "No arXiv papers found matching" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_interactive_loop_exit_command(self, mock_pipeline, mock_graph):
        mock_pipeline.return_value = {
            "status": "summarized",
            "selected_paper": SAMPLE_PAPER,
            "briefing": SAMPLE_BRIEFING,
            "candidate_papers": [SAMPLE_PAPER],
        }

        result = runner.invoke(app, ["--query", "1706.03762"], input="exit\n")
        assert result.exit_code == 0
        assert "Interactive QA Session Active" in result.stdout
        assert "Goodbye" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_interactive_loop_help_command(self, mock_pipeline, mock_graph):
        mock_pipeline.return_value = {
            "status": "summarized",
            "selected_paper": SAMPLE_PAPER,
            "briefing": SAMPLE_BRIEFING,
            "candidate_papers": [SAMPLE_PAPER],
        }

        result = runner.invoke(app, ["--query", "1706.03762"], input="help\nexit\n")
        assert result.exit_code == 0
        assert "Available Commands" in result.stdout
        assert "exit / quit" in result.stdout

    @patch("src.cli.main.build_agent_graph")
    @patch("src.cli.main.run_ingestion_pipeline")
    def test_cli_interactive_loop_status_command(self, mock_pipeline, mock_graph):
        mock_pipeline.return_value = {
            "status": "summarized",
            "selected_paper": SAMPLE_PAPER,
            "briefing": SAMPLE_BRIEFING,
            "candidate_papers": [SAMPLE_PAPER],
        }

        result = runner.invoke(app, ["--query", "1706.03762", "--session-id", "test_sess_100"], input="status\nexit\n")
        assert result.exit_code == 0
        assert "Session Status" in result.stdout
        assert "test_sess_100" in result.stdout
