"""Offline verification and analysis of recorded LLM responses."""

import argparse
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

from .analysis import analyze
from .evidence import load_bundle, read_csv


def verify_reference(output, reference):
    comparisons = 0
    max_difference = 0.0
    for name in (
        "model_summary.csv",
        "paired_accuracy.csv",
        "policy_summary.csv",
        "policy_comparisons_vs_raw.csv",
        "failures.csv",
    ):
        actual, expected = read_csv(output / name), read_csv(reference / name)
        if len(actual) != len(expected):
            raise ValueError("Reference row count differs: " + name)
        for first, second in zip(actual, expected, strict=True):
            if first.keys() != second.keys():
                raise ValueError("Reference columns differ: " + name)
            for field in first:
                if first[field] == second[field]:
                    continue
                try:
                    a, b = float(first[field]), float(second[field])
                except ValueError as error:
                    raise ValueError(f"Reference differs: {name}:{field}") from error
                if not np.isclose(a, b, rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Reference differs: {name}:{field}")
                max_difference = max(max_difference, abs(a - b))
            comparisons += 1
    return {"reference_rows_compared": comparisons, "max_numeric_difference": max_difference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check hashes, tasks, grades, fees and reference analysis")
    run = commands.add_parser("analyze", help="Recompute all results without calling models")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs, tasks, rows = load_bundle(args.root)
    if args.command == "analyze":
        result = analyze(args.root, args.output, configs, tasks, rows)
        (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        with tempfile.TemporaryDirectory(prefix="fahrai_llm_") as folder:
            output = Path(folder) / "analysis"
            result = analyze(args.root, output, configs, tasks, rows)
            result.update(verify_reference(output, args.root / "data/llm/reference"))
        result["status"] = "PASS"
    result["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    print(json.dumps(result, indent=2))
