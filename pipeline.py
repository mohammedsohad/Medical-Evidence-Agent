"""
Medical Evidence Agent - core pipeline.

This module contains the shared multi-agent pipeline logic (Search -> Extract
-> Synthesize -> Guardrail), extracted into an importable file so it can be
reused by both `cli.py` (Agent CLI skill) and `mcp_server.py` (MCP Server),
instead of duplicating the pipeline in each entry point.
"""

import json
import os
import re
import time

import requests
import xml.etree.ElementTree as ET

from google import genai
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY from a local .env file, if present


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def extract_json(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Security Layer
# ---------------------------------------------------------------------------
class SecurityError(Exception):
    """Raised when user-supplied input fails validation before hitting an API call."""
    pass


MAX_QUESTION_LENGTH = 300

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |all )?(system|above) prompt",
    r"you are now (a|an) ",
    r"new instructions:",
    r"reveal (your|the) (system prompt|api key)",
]


def validate_research_question(question):
    """Validate user-supplied input before it is used to build prompts or API calls."""
    if not isinstance(question, str) or not question.strip():
        raise SecurityError("Research question must be a non-empty string.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise SecurityError(
            "Research question exceeds " + str(MAX_QUESTION_LENGTH) + " characters."
        )
    return question.strip()


def sanitize_external_text(text, max_len=4000):
    """Neutralize likely prompt-injection attempts in text pulled from an external
    source (PubMed) before it is embedded into an LLM prompt."""
    if not isinstance(text, str):
        return "not_reported"
    cleaned = text[:max_len]
    for pattern in PROMPT_INJECTION_PATTERNS:
        cleaned = re.sub(pattern, "[filtered]", cleaned, flags=re.IGNORECASE)
    return cleaned


def redact_secret(value):
    """Return a masked preview of a secret for safe logging. Never log a full key."""
    if not value:
        return "<missing>"
    return value[:4] + "..." + "*" * 4


def load_gemini_api_key():
    """Load the Gemini API key without ever hardcoding or printing it.

    Resolution order:
    1. Kaggle Secrets (Add-ons > Secrets) - used when running on Kaggle.
    2. GEMINI_API_KEY environment variable - used for local/self-hosted runs,
       e.g. set via a `.env` file (see `.env.example`).
    """
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. On Kaggle: Add-ons > Secrets. "
            "Locally: set the GEMINI_API_KEY environment variable or add it to a .env file."
        )
    return key


GEMINI_API_KEY = load_gemini_api_key()
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


def generate_with_retry(prompt, model_name=MODEL_NAME, max_retries=6, base_delay=15):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = base_delay * (attempt + 1)
                print("Rate limit hit, waiting " + str(wait_time) + "s...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Failed after max retries due to rate limiting.")


# ---------------------------------------------------------------------------
# PubMed data source
# ---------------------------------------------------------------------------
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query, max_results=1):
    time.sleep(1)
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    resp = requests.get(PUBMED_ESEARCH, params=params, timeout=15)
    resp.raise_for_status()
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])

    results = []
    for pmid in id_list:
        time.sleep(1)
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "xml",
        }
        fresp = requests.get(PUBMED_EFETCH, params=fetch_params, timeout=15)
        fresp.raise_for_status()
        root = ET.fromstring(fresp.content)

        title_el = root.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else "not_reported"

        year_el = root.find(".//PubDate/Year")
        pub_year = year_el.text if year_el is not None else "not_reported"

        abstract_parts = [el.text for el in root.findall(".//AbstractText") if el.text]
        abstract = " ".join(abstract_parts) if abstract_parts else "not_reported"

        results.append({"pmid": pmid, "title": title, "abstract": abstract, "pub_year": pub_year})

    return results


# ---------------------------------------------------------------------------
# Agent 1: Search Unit
# ---------------------------------------------------------------------------
class KnowledgeRetrievalAgent:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def build_search_queries(self, research_question):
        prompt_lines = [
            "You are an academic research assistant specialized in Medical Evidence.",
            "Convert the following research question into 5 precise scientific search queries",
            "in English, suitable for databases like PubMed.",
            "",
            "Research question: " + research_question,
            "",
            "Return ONLY valid JSON in this exact format:",
            '{"queries": ["query1", "query2", "query3", "query4", "query5"]}',
        ]
        prompt = "\n".join(prompt_lines)
        response = generate_with_retry(prompt, self.model_name)
        try:
            data = extract_json(response.text)
            return data.get("queries", [])
        except (json.JSONDecodeError, ValueError):
            return [response.text.strip()]

    def get_documents(self, text_input):
        queries = self.build_search_queries(text_input)
        documents = []
        doc_id = 1
        for q in queries:
            pubmed_results = search_pubmed(q, max_results=1)
            for r in pubmed_results:
                raw_text = r["title"] + ". " + r["abstract"]
                combined_text = sanitize_external_text(raw_text)
                documents.append({
                    "id": doc_id,
                    "query_used": q,
                    "pmid": r["pmid"],
                    "pub_year": r.get("pub_year", "not_reported"),
                    "text": combined_text,
                })
                doc_id += 1
        return documents


# ---------------------------------------------------------------------------
# Agent 2: Data Extractor
# ---------------------------------------------------------------------------
class FeatureExtractionAgent:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def process_data(self, dataset):
        results = []
        for doc in dataset:
            time.sleep(13)
            prompt_lines = [
                "You are a biomedical data extraction assistant.",
                "From the study metadata below, extract structured features.",
                "",
                "Study metadata: " + str(doc.get("text", "")),
                "Related search query: " + str(doc.get("query_used", "")),
                "",
                "Return ONLY valid JSON in this exact format:",
                '{"sample_size": "...", "study_design": "...", "key_statistics": "...", "validated": true}',
                "If information is not available in the text, use the string not_reported for that field.",
            ]
            prompt = "\n".join(prompt_lines)

            response = generate_with_retry(prompt, self.model_name)

            try:
                extracted = extract_json(response.text)
            except (json.JSONDecodeError, ValueError):
                extracted = {
                    "sample_size": "not_reported",
                    "study_design": "not_reported",
                    "key_statistics": "not_reported",
                    "validated": False,
                }

            extracted["study_id"] = doc.get("id")
            extracted["pub_year"] = doc.get("pub_year", "not_reported")
            results.append(extracted)

        return results


# ---------------------------------------------------------------------------
# Agent 3: Synthesizer Unit
# ---------------------------------------------------------------------------
class InformationSynthesizerAgent:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name

    def output_summary(self, processed_metrics):
        prompt_lines = [
            "You are a medical evidence synthesis assistant.",
            "Below is a list of extracted features from multiple clinical studies.",
            "Write a comprehensive academic summary in English that:",
            "- States the overall evidence trend across the studies.",
            "- Notes any limitations (e.g. missing sample size or study design data).",
            "- Avoids inventing any numbers or facts not present in the data.",
            "",
            "Extracted data: " + str(processed_metrics),
            "",
            "Return ONLY valid JSON in this exact format:",
            '{"summary": "...", "studies_reviewed": 0, "limitations_noted": "..."}',
        ]
        prompt = "\n".join(prompt_lines)

        response = generate_with_retry(prompt, self.model_name)

        try:
            data = extract_json(response.text)
        except (json.JSONDecodeError, ValueError):
            data = {
                "summary": response.text.strip(),
                "studies_reviewed": len(processed_metrics),
                "limitations_noted": "not_reported",
            }

        return data


# ---------------------------------------------------------------------------
# Guardrail Validator
# ---------------------------------------------------------------------------
class GuardrailValidator:
    def __init__(self):
        self.warnings = []

    def validate_pipeline_output(self, documents, processed_metrics, final_summary):
        self.warnings = []

        if len(documents) != len(processed_metrics):
            self.warnings.append(
                "Mismatch: " + str(len(documents)) + " documents retrieved but "
                + str(len(processed_metrics)) + " studies processed."
            )

        claimed_count = final_summary.get("studies_reviewed", None)
        actual_count = len(processed_metrics)
        if claimed_count != actual_count:
            self.warnings.append(
                "Synthesizer claimed " + str(claimed_count) + " studies reviewed, "
                "but " + str(actual_count) + " were actually processed."
            )

        required_keys = ["summary", "studies_reviewed", "limitations_noted"]
        for key in required_keys:
            if key not in final_summary:
                self.warnings.append("Missing required field in summary: " + key)

        all_sample_sizes_missing = all(
            str(m.get("sample_size", "")).strip().lower() == "not_reported"
            for m in processed_metrics
        )
        summary_text = str(final_summary.get("summary", "")).lower()
        if all_sample_sizes_missing and "total of" in summary_text:
            self.warnings.append(
                "Possible hallucination: summary references a total sample size "
                "even though no individual study reported one."
            )

        for m in processed_metrics:
            if "study_id" not in m:
                self.warnings.append("A processed study is missing its study_id (not traceable to source).")

        is_valid = len(self.warnings) == 0
        return {"is_valid": is_valid, "warnings": self.warnings}


# ---------------------------------------------------------------------------
# Custom Skills
# ---------------------------------------------------------------------------
class CustomSkills:
    @staticmethod
    def rank_by_sample_size(processed_metrics):
        def extract_number(value):
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return int(digits) if digits else -1

        return sorted(
            processed_metrics,
            key=lambda m: extract_number(m.get("sample_size", "")),
            reverse=True,
        )

    @staticmethod
    def rank_by_publication_date(processed_metrics):
        def extract_year(value):
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return int(digits) if len(digits) == 4 else -1

        return sorted(
            processed_metrics,
            key=lambda m: extract_year(m.get("pub_year", "")),
            reverse=True,
        )

    @staticmethod
    def filter_validated_studies(processed_metrics):
        return [m for m in processed_metrics if m.get("validated") is True]

    @staticmethod
    def classify_evidence_quality(processed_metrics):
        classified = []
        for m in processed_metrics:
            sample_size_known = str(m.get("sample_size", "")).strip().lower() != "not_reported"
            design_known = str(m.get("study_design", "")).strip().lower() != "not_reported"

            if sample_size_known and design_known:
                quality = "high_traceability"
            elif sample_size_known or design_known:
                quality = "partial_traceability"
            else:
                quality = "low_traceability"

            entry = dict(m)
            entry["evidence_quality"] = quality
            classified.append(entry)
        return classified


# ---------------------------------------------------------------------------
# Shared pipeline entry point (used by cli.py and mcp_server.py)
# ---------------------------------------------------------------------------
def run_pipeline(research_question):
    question = validate_research_question(research_question)

    agent1 = KnowledgeRetrievalAgent()
    documents = agent1.get_documents(question)

    agent2 = FeatureExtractionAgent()
    processed = agent2.process_data(documents)

    agent3 = InformationSynthesizerAgent()
    summary = agent3.output_summary(processed)

    validator = GuardrailValidator()
    validation = validator.validate_pipeline_output(documents, processed, summary)

    return {
        "summary": summary,
        "guardrail": validation,
        "studies_processed": len(processed),
    }
