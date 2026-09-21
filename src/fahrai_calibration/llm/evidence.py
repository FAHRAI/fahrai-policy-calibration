"""Read and validate the published experiment bundle."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .grading import grade, parse_response
from .tasks import INSTRUCTION, SCHEMA, canonical, generate, schedule


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def check_requests(requests, tasks, configs):
    expected = {"exercise", "reference"}
    for row in requests:
        task = tasks[row["assignment_id"].rsplit(":r", 1)[0]]
        body = row["request"]
        provider = configs[row["configuration"]]["provider"]
        if provider == "openai":
            instruction = body["input"][0]["content"]
            payload = body["input"][1]["content"]
            schema = body["text"]["format"]["schema"]
        elif provider == "anthropic":
            instruction = body["system"]
            payload = body["messages"][0]["content"]
            schema = body["output_config"]["format"]["schema"]
        elif provider == "gemini":
            instruction = body["systemInstruction"]["parts"][0]["text"]
            payload = body["contents"][0]["parts"][0]["text"]
            schema = body["generationConfig"]["responseJsonSchema"]
        else:
            instruction = body["messages"][0]["content"]
            payload = body["messages"][1]["content"]
            schema = body["format"]
        data = json.loads(payload)
        if (
            instruction != INSTRUCTION
            or schema != SCHEMA
            or set(data) != expected
            or data != {key: task[key] for key in expected}
        ):
            raise ValueError("Request content changed or exposes evaluation labels")


def load_bundle(root, checksums=True):
    folder = root / "data/llm"
    if checksums:
        for name, expected in read_json(folder / "SHA256.json").items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Checksum mismatch: " + name)
    configs = read_json(folder / "configurations.json")
    tasks = read_json(folder / "tasks.json")
    if generate(read_csv(root / "data/population.csv")) != tasks:
        raise ValueError("Task generation or reference answers differ")
    assignments = read_json(folder / "assignments.json")
    if schedule(tasks) != assignments:
        raise ValueError("Assignment schedule differs")
    by_id = {task["task_id"]: task for task in tasks}
    requests = read_jsonl(folder / "requests.jsonl")
    check_requests(requests, by_id, configs)
    request_map = {(r["configuration"], r["assignment_id"]): r["request"] for r in requests}
    expected = {
        (c, j["assignment_id"]) for c in configs for j in assignments if j["phase"] == "main"
    }
    if set(request_map) != expected or len(requests) != len(expected):
        raise ValueError("Missing or duplicate requests")
    rows = read_jsonl(folder / "responses.jsonl")
    if Counter((r["configuration"], r["assignment_id"]) for r in rows) != Counter(expected):
        raise ValueError("Missing or duplicate responses")
    for row in rows:
        task = by_id[row["task_id"]]
        key = (row["configuration"], row["assignment_id"])
        if row["phase"] != "main" or task["phase"] != "main":
            raise ValueError("Development task in main analysis")
        if row["assignment_id"] != f"{row['task_id']}:r{row['repeat']}":
            raise ValueError("Incorrect assignment")
        if digest(request_map[key]) != row["request_sha256"]:
            raise ValueError("Request hash differs")
        if hashlib.sha256(row["response_text"].encode()).hexdigest() != row["response_sha256"]:
            raise ValueError("Response text hash differs")
        parsed = parse_response(row)
        if parsed != row["parsed_response"]:
            raise ValueError("Saved parsed response differs from final text")
        for field, value in grade(parsed, task).items():
            if row[field] != value:
                raise ValueError("Recorded grade differs: " + field)
        for field in ("latency_s", "api_fee_estimate_usd"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("Invalid measurement: " + field)
        conf = configs[row["configuration"]]
        if row["provider"] != conf["provider"] or row["model_requested"] != conf["model"]:
            raise ValueError("Configuration differs")
        if row["provider"] == "ollama":
            if row["api_fee_estimate_usd"] != 0 or row["total_compute_cost_usd"] is not None:
                raise ValueError("Local fee must not be treated as total compute cost")
        else:
            usage = row["usage"]
            estimate = (
                usage["input_tokens"] * conf["input_usd_per_million"]
                + usage["output_tokens"] * conf["output_usd_per_million"]
                + usage["cache_creation_input_tokens"] * 0.25 * conf["input_usd_per_million"]
            ) / 1e6
            if not math.isclose(
                estimate, row["api_fee_estimate_usd"], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("API fee estimate differs")
    return configs, by_id, rows
