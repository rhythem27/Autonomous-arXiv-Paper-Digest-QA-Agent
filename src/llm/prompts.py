"""Prompt templates and formatting utilities for Executive Briefing and Grounded QA.

Enforces strict structural constraints for paper digestion, mandatory limitations,
and anti-hallucination guardrails during question answering.
"""

from typing import Dict, Any, List


ANTI_HALLUCINATION_REFUSAL_PREFIX = "The provided text from this paper does not contain information regarding"

BRIEFING_SYSTEM_PROMPT = """You are an expert AI research scientist and technical reviewer.
Your mission is to read the provided structured research paper excerpts and produce a rigorous, authoritative Executive Briefing.

You MUST adhere strictly to the following 7 required sections:
1. METADATA: Title, authors, arXiv ID, publish date, link.
2. WHY IT MATTERS: Exactly 1 cohesive paragraph written in plain English explaining why this research matters to machine learning practitioners, engineers, and researchers.
3. PROBLEM STATEMENT: The precise technical bottleneck, theoretical gap, or practical limitation the paper aims to solve.
4. METHOD / APPROACH: A bulleted breakdown of the core architectural novelty, algorithmic components, training regime, or theoretical proofs.
5. KEY RESULTS: Specific quantitative metrics, benchmark scores, speedups, ablation insights, or theoretical claims established by the authors.
6. LIMITATIONS (STRICT NEGATIVE CONSTRAINT):
   - You MUST NOT omit, skip, or gloss over this section.
   - You MUST NOT provide vague generalities like "more work is needed".
   - You MUST explicitly detail the stated assumptions, dataset limitations, hardware/compute bottlenecks, failure modes, degradation under distribution shifts, or unaddressed edge cases.
7. SUGGESTED FOLLOW-UP QUESTIONS: 3 to 5 insightful, critical questions that a senior researcher or peer reviewer would ask the authors.

You must respond ONLY with a valid, parseable JSON object matching this exact JSON schema:
{
  "why_it_matters": "...",
  "problem_statement": "...",
  "method_approach": ["point 1", "point 2", ...],
  "key_results": ["claim 1 with numbers", "claim 2 with numbers", ...],
  "limitations": ["explicit limitation 1", "explicit limitation 2", ...],
  "suggested_follow_ups": ["question 1?", "question 2?", ...]
}
"""

BRIEFING_USER_PROMPT_TEMPLATE = """Please generate an Executive Briefing for the following research paper:

TITLE: {title}
AUTHORS: {authors}
ARXIV ID: {arxiv_id}
PUBLISHED: {published}
LINK: {link}

=== PAPER EXCERPTS ===
ABSTRACT:
{abstract}

KEY SECTIONS:
{sections_text}
======================

Output ONLY the JSON object. Do not include markdown codeblocks or other formatting."""


GROUNDED_QA_SYSTEM_PROMPT = """You are a precise scientific research assistant answering questions about a specific research paper.
Base your answers SOLELY on the retrieved excerpts provided below.

STRICT GROUNDING RULES:
1. If the information needed to answer the question is NOT present in or cannot be directly deduced from the provided excerpts, you MUST state:
   "The provided text from this paper does not contain information regarding [topic]."
   Do NOT guess, extrapolate, or introduce external training facts not confirmed by the excerpts.
2. Always cite the exact section names you referenced when formulating your answer using the tag format: [Section: <Section Name>].
3. Be clear, technically precise, and concise.
"""

GROUNDED_QA_USER_PROMPT_TEMPLATE = """=== RETRIEVED PAPER EXCERPTS ===
{context}
================================

USER QUESTION:
{question}

Provide a grounded, truthful answer citing the relevant sections. If the paper excerpts do not contain the answer, decline according to Grounding Rule 1."""


def format_briefing_markdown(briefing: Dict[str, Any]) -> str:
    """Format an ExecutiveBriefing dictionary into clean, presentation-ready Markdown."""
    authors_str = ", ".join(briefing.get("authors", []))
    method_bullets = "\n".join(f"- {m}" for m in briefing.get("method_approach", []))
    results_bullets = "\n".join(f"- {r}" for r in briefing.get("key_results", []))
    limitations_bullets = "\n".join(f"- {l}" for l in briefing.get("limitations", []))
    follow_up_bullets = "\n".join(f"- {q}" for q in briefing.get("suggested_follow_ups", []))

    return f"""# Executive Briefing: {briefing.get('title', 'Untitled Paper')}

**arXiv ID**: `{briefing.get('arxiv_id', 'N/A')}` | **Published**: {briefing.get('publish_date', 'N/A')}  
**Authors**: {authors_str}  
**Official Link**: {briefing.get('link', 'N/A')}

---

## 1. Why This Paper Matters
{briefing.get('why_it_matters', 'N/A')}

## 2. Problem Statement
{briefing.get('problem_statement', 'N/A')}

## 3. Core Method & Technical Approach
{method_bullets}

## 4. Key Results & Empirical Claims
{results_bullets}

## 5. Limitations & Bottlenecks (Critical Assessment)
{limitations_bullets}

## 6. Suggested Follow-Up Questions
{follow_up_bullets}
"""
