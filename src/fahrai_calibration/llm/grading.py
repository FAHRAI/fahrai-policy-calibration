"""Reproduce the recorded grading rule without an LLM judge."""

import json

COMPLETE = {"openai": "completed", "anthropic": "end_turn", "gemini": "STOP", "ollama": "stop"}


def parse_response(row):
    if row["finish_reason"] != COMPLETE[row["provider"]]:
        return None
    try:
        return json.loads(row["response_text"])
    except (ValueError, TypeError):
        return None


def grade(parsed, task):
    valid = (
        isinstance(parsed, dict)
        and set(parsed) == {"final_answer", "explanation", "source_ids"}
        and type(parsed.get("final_answer")) is int
        and isinstance(parsed.get("explanation"), str)
        and bool(parsed["explanation"].strip())
        and isinstance(parsed.get("source_ids"), list)
        and all(isinstance(source, str) for source in parsed["source_ids"])
    )
    return {
        "schema_valid": bool(valid),
        "answer_exact": bool(valid and parsed["final_answer"] == task["gold_answer"]),
        "source_ids_valid": bool(
            valid
            and parsed["source_ids"]
            and set(parsed["source_ids"]) == {task["reference"]["source_id"]}
        ),
    }
