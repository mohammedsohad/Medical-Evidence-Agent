"""
Agent CLI Skill.

Usage:
    python cli.py "What is the effectiveness of GLP-1 receptor agonists in weight loss?"
    python cli.py "..." --json
"""

import argparse
import json
import sys

from pipeline import run_pipeline, SecurityError


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="medical-evidence-agent",
        description="Run the multi-agent medical evidence review pipeline from the command line.",
    )
    parser.add_argument("question", help="The medical research question to investigate.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(args.question)
    except SecurityError as e:
        print("Input rejected: " + str(e), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n=== Medical Evidence Summary ===")
        print(result["summary"].get("summary", "not_reported"))
        print("\nStudies reviewed: " + str(result["studies_processed"]))
        print("Guardrail valid: " + str(result["guardrail"]["is_valid"]))
        if not result["guardrail"]["is_valid"]:
            print("Warnings:")
            for w in result["guardrail"]["warnings"]:
                print("- " + w)

    return result


if __name__ == "__main__":
    main()
