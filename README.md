# Medical Evidence Agent
### An automated multi-agent system that turns a medical research question into a sourced, guardrail-checked evidence summary

---

## Problem Statement

As a doctor, I see firsthand how medical researchers and frontline clinicians are completely overwhelmed by the exponential growth of scientific literature coming out every day. Staying up-to-date on a specific treatment means hours spent digging through PubMed, scanning dozens of abstracts, and trying to mentally calculate sample sizes, study designs, and statistical outcomes across conflicting papers. It is exhausting, prone to human error, and simply doesn't scale with the speed of modern medical research.

Naturally, large language models (LLMs) seem like the perfect solution to speed this up. But throwing a general "one-size-fits-all" prompt at a standard LLM introduces a much more dangerous problem: silent hallucinations. If you ask a native LLM to "summarize the evidence for drug X," it will often give you a beautifully fluent, highly confident response that quietly invents a sample size, a p-value, or a clinical conclusion that doesn't exist in any actual paper. In medicine, that isn't just a cosmetic bug — it's a dangerous flaw that breaks clinical trust.

I built Medical Evidence Agent to bridge this exact gap: automating the literature review process while introducing a structural, multi-layered defense that makes it incredibly hard for the system to fabricate evidence.

---

## Why Agents? (The ADK Philosophy)

When designing this project, I deliberately rejected the "one big prompt" approach. Instead, I leaned fully into the **Agent Development Kit (ADK)** philosophy taught in this course: breaking down a complex workflow into specialized, single-purpose agents with strict inputs and outputs, wiring them together in a clean pipeline, and auditing the final output with an independent validation layer that is entirely non-LLM.

This architecture matters for three practical reasons:

- **Isolating Errors:** A single model trying to search, extract, and write a summary all at once has to juggle too many cognitive tasks. A mistake in retrieval instantly ruins the extraction. By splitting this into three sub-agents, each one has exactly one job, one clear prompt, and one defined contract.
- **Hyper-Focused Prompts:** The extraction agent's entire existence is built around one rule: force missing data to be labeled as `not_reported` instead of allowing the model to guess. A prompt with this narrow of a scope is vastly more reliable than a general summarization prompt.
- **Independent Auditing:** Because each agent outputs a discrete, inspectable artifact (a clean list of documents, a structured JSON of metrics, a narrative summary), a final programmatic Guardrail layer can audit the entire chain. This would be completely impossible inside one opaque, single-prompt model call.

Ultimately, using agents isn't a buzzword here — it is the exact engineering mechanism that makes LLM hallucinations visible and catchable.

---

## System Architecture

```
Research Question
        │
        ▼
[KnowledgeRetrievalAgent] ──▶ PubMed (Live E-utilities API)
        │
        ▼
   Real Documents (Published papers — strictly no synthetic data)
        │
        ▼
[FeatureExtractionAgent] ──▶ Gemini (Structured feature parsing)
        │
        ▼
   Processed Studies (Metrics mapped with 'not_reported' safety flags)
        │
        ▼
[InformationSynthesizerAgent] ──▶ Gemini (Rigorous academic synthesis)
        │
        ▼
[GuardrailValidator] ──▶ Deterministic consistency & validation check
        │
        ▼
   Final Audited Summary
```

**Agent 1 — KnowledgeRetrievalAgent (Search Unit):** This unit takes a clinician's natural language question and converts it into five highly targeted, scientific search queries. It then queries PubMed live via NCBI's E-utilities API. Grounding our pipeline in actual, published literature — rather than letting an LLM recall papers from its weights — is our first line of defense against hallucination.

**Agent 2 — FeatureExtractionAgent (Data Extractor):** For every paper retrieved, this agent uses Gemini to extract structured fields: sample size, study design, and key statistics. The prompt is heavily constrained: if a metric isn't explicitly in the text, it must output `not_reported`. This is our second line of defense.

**Agent 3 — InformationSynthesizerAgent (Synthesizer Unit):** This agent takes the structured JSON data from all evaluated studies and weaves them into a single, cohesive academic overview, explicitly highlighting data gaps and limitations rather than inventing facts to smooth over the report.

**GuardrailValidator (The Auditor):** This is a completely deterministic, independent Python validation layer (not an LLM call). It acts as a hard gatekeeper by checking internal consistency: Does the number of papers found match the number processed? Does the synthesizer's claimed study count match reality? Most importantly, it scans for phrases like "a total of X patients" if individual data was missing, catching pooled hallucinations before a doctor ever sees them.

---

## Security Features

Medical data demands strict boundaries. I built a lightweight, dependency-free security layer directly into the core execution pipeline handling two main risks:

- **Untrusted User Input:** The initial research question is strictly checked for length and type validation via a `validate_research_question` function, raising a dedicated `SecurityError` if anything looks malicious or malformed.
- **Prompt Injection Guardrails:** Since we fetch live text from third-party studies on PubMed, we treat that external data as untrusted. The text is passed through an injection filter looking for phrases like "ignore previous instructions" or "reveal system prompt" before being embedded into Gemini's prompt context.

Additionally, API keys are completely protected: the Gemini key is dynamically resolved using Kaggle Secrets in the notebook environment, and any system logs only print a redacted preview (`sk-a...****`) to prevent accidental leaks.

---

## Agent Skills — The CLI Tool

To prove this architecture is robust and ready for production, I wrapped the entire agent pipeline logic into a clean command-line interface (`cli.py`). This allows researchers to run full, guardrail-checked literature reviews directly from their terminal without opening a notebook:

```bash
python cli.py "What is the effectiveness of GLP-1 receptor agonists in weight loss?"
python cli.py "..." --json
```

The CLI imports the exact same shared `pipeline.py` module used by the notebook, meaning the core logic, security rules, and agent definitions remain perfectly unified without code duplication.

---

## MCP Server Integration

To make this system truly interoperable, the multi-agent pipeline is exposed as a set of Model Context Protocol (MCP) tools via `mcp_server.py`. This allows any modern MCP client — like Claude Desktop or an enterprise orchestration tool — to invoke our agent natively.

Rather than exposing one massive tool, I split the functionality into three discrete capabilities:

1. **`search_medical_literature`**: Executes the entire end-to-end Search → Extract → Synthesize → Guardrail loop.
2. **`rank_studies_by_sample_size`**: A deterministic skill that filters and orders already-extracted studies by numerical patient enrollment.
3. **`classify_studies_by_evidence_quality`**: Classifies processed studies into clear traceability tiers (`high_traceability`, `partial_traceability`, `low_traceability`) based on reporting completeness.

This modular design allows an external LLM coordinator to call the main pipeline once, and then flexibly re-rank or re-classify the cached results in multiple ways without wasting API tokens or re-querying PubMed.

---

## The Build

- **Language / runtime:** Python 3.10+
- **LLM Integration:** Google Gemini (`gemini-2.5-flash`) utilizing the official `google-genai` SDK. It includes a custom retry wrapper that catches rate-limit errors (`RESOURCE_EXHAUSTED` / `429`), applies exponential backoff, and resumes execution seamlessly.
- **Data Source:** Live PubMed (NCBI) E-utilities API.
- **Protocol:** Model Context Protocol (`mcp` / `FastMCP` ecosystem).
- **Secrets Management:** `python-dotenv` for local environments, Kaggle Secrets for the notebook.

---

## Repository Structure

| File | Purpose |
|---|---|
| `notebook.ipynb` | The fully annotated, step-by-step interactive notebook (primary deliverable) |
| `pipeline.py` | The standalone, production-ready module containing the core agent classes and security filters |
| `cli.py` | The command-line tool skill to execute the pipeline from any terminal |
| `mcp_server.py` | The MCP server code mapping our system and skills as composable tools |
| `requirements.txt` | Python dependencies |
| `.env.example` | Local environment configuration template |

---

## Setup Instructions

### 1. On Kaggle (as submitted)
1. Open the notebook in a Kaggle Notebook.
2. Add your Gemini API key under **Add-ons → Secrets** with the name `GEMINI_API_KEY`.
3. Enable internet access for the notebook (required for the PubMed API and the Gemini API).
4. Run all cells top to bottom.

### 2. Locally / self-hosted
```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=your_real_key
```

Run the CLI:
```bash
python cli.py "What is the effectiveness of GLP-1 receptor agonists in weight loss?"
python cli.py "..." --json
```

Run the MCP server (for use with an MCP-compatible client such as Claude Desktop):
```bash
python mcp_server.py
```

---

## Live Demo & Experience

When testing the pipeline with the clinical question: *"What is the effectiveness of GLP-1 receptor agonists in weight loss?"*, the interaction behaves flawlessly across all entry points (Notebook, CLI, and MCP). The Search agent formats queries, the Fetch tool grabs live medical papers, the Extractor normalizes the metrics, and the Synthesizer outputs a highly structured clinical summary. Right after, the Guardrail layer logs its audit trail, printing the exact number of verified studies and explicit warning flags if data gaps were caught.

---

## Limitations & Future Scope

- **Abstract Dependency:** The data extraction naturally depends on what is reported in the PubMed abstracts. If an abstract is heavily compressed, the agent will flag it as `not_reported`. While our guardrails catch these omissions, the system is an aid, not a replacement for full-text expert clinical review.
- **API Quotas:** To remain fully compliant with the free-tier rate limits of both PubMed and Gemini, the pipeline currently samples one key study per generated query path. Increasing this count would expand literature breadth in a production environment.

---

## Closing Thoughts

Developing the Medical Evidence Agent showed me that a multi-agent architecture isn't just an elegant way to organize code — it is what makes LLMs safe enough to use in high-stakes fields like medicine. By forcing retrieval, extraction, synthesis, and validation into separate, auditable slots, and exposing that logic across a notebook, a CLI, and an MCP server, this project shows how the ADK philosophy can be turned into a practical, highly reusable tool for the scientific community.
