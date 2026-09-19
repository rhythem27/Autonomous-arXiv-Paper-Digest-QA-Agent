import ast
import json
from pathlib import Path
import pytest

NOTEBOOK_PATH = Path("demo.ipynb")


def test_notebook_exists_and_valid_json():
    """Verify demo.ipynb exists and is valid JSON matching nbformat 4."""
    assert NOTEBOOK_PATH.exists(), "demo.ipynb does not exist"
    
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    assert "cells" in nb
    assert "metadata" in nb
    assert nb.get("nbformat") == 4
    assert len(nb["cells"]) >= 7


def test_notebook_code_cells_syntax():
    """Verify that all code cells in demo.ipynb parse cleanly with ast.parse."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert len(code_cells) >= 7, f"Expected at least 7 code cells, found {len(code_cells)}"

    for idx, cell in enumerate(code_cells):
        code_source = "".join(cell.get("source", []))
        try:
            ast.parse(code_source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in code cell {idx}: {e}\nCode:\n{code_source}")


def test_notebook_required_sections_present():
    """Verify that all key task criteria are present in notebook cells."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_sources = "".join("".join(c.get("source", [])) for c in nb["cells"])

    # 1. Environment & setup
    assert "build_agent_graph" in all_sources
    assert "settings" in all_sources
    
    # 2. State graph visualization
    assert "draw_mermaid" in all_sources
    
    # 3. Stream execution
    assert "graph.stream" in all_sources
    
    # 4. Candidate table / ranking
    assert "candidate_papers" in all_sources or "selected_paper" in all_sources
    
    # 5. PyMuPDF sections & Qdrant local
    assert "QdrantStore" in all_sources
    assert "search_chunks" in all_sources
    
    # 6. Executive briefing
    assert "briefing" in all_sources
    assert "markdown_output" in all_sources
    
    # 7. Grounded QA & anti-hallucination & ipywidgets
    assert "does not contain information" in all_sources
    assert "widgets.Text" in all_sources
    assert "widgets.Button" in all_sources
    assert "widgets.Output" in all_sources
