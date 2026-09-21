"""Offline verification and analysis of recorded LLM responses."""

import argparse
import json
import platform
import tempfile
from pathlib import Path

import numpy as np

from .analysis import analyze
from .verification import load_bundle, verify_reference


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check hashes, tasks, grades, fees and reference analysis")
    run = commands.add_parser("analyze", help="Recompute all results without calling models")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    configs, tasks, rows = load_bundle(args.root)
    if args.command == "analyze":
        result = analyze(args.root, args.output, configs, tasks, rows)
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
    report = json.dumps(result, indent=2) + "\n"
    if args.command == "analyze":
        (args.output / "validation.json").write_text(report, encoding="utf-8")
    print(report, end="")
