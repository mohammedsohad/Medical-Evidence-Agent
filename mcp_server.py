"""
MCP Server for the Medical Evidence Agent.

Exposes the multi-agent pipeline as Model Context Protocol tools so any
MCP-compatible client (Claude Desktop, another agent, a custom orchestrator)
can call it directly.

Run it with:
    python mcp_server.py

Then point an MCP client at this process (stdio transport by default).
"""

from mcp.server.fastmcp import FastMCP

from pipeline import (
    run_pipeline,
    CustomSkills,
)

mcp_server = FastMCP("medical-evidence-agent")


@mcp_server.tool()
def search_medical_literature(research_question: str) -> dict:
    """Run the full Search -> Extract -> Synthesize pipeline for a medical
    research question and return the validated, guardrail-checked summary."""
    return run_pipeline(research_question)


@mcp_server.tool()
def rank_studies_by_sample_size(processed_metrics: list) -> list:
    """Rank already-extracted studies by reported sample size, largest first."""
    return CustomSkills.rank_by_sample_size(processed_metrics)


@mcp_server.tool()
def classify_studies_by_evidence_quality(processed_metrics: list) -> list:
    """Classify already-extracted studies by traceability of their reported evidence."""
    return CustomSkills.classify_evidence_quality(processed_metrics)


if __name__ == "__main__":
    mcp_server.run()
