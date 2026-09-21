"""Offline summaries, paired task bootstrap, and policy replay."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from .evidence import read_csv, write_csv

METRICS = (
    "calls",
    "missed_error_mass",
    "correct_executions_per_case",
    "failed_executions_per_case",
    "api_fee_estimate_usd_per_case",
    "sequential_seconds_per_case",
)
THRESHOLDS = (0.4, 0.6, 0.7, 0.8, 0.9)
BUDGETS = tuple(i / 10 for i in range(11))
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_DRAWS = 2000


def budget_route(scores, masses, budget):
    """Allocate target mass by score, randomizing uniformly within tied scores."""
    if not 0 <= budget <= 1:
        raise ValueError("Budget must lie in [0, 1]")
    groups = defaultdict(list)
    for cell, score in enumerate(scores):
        groups[score].append(cell)
    route = np.zeros(len(scores))
    remaining = budget
    for score in sorted(groups):
        cells = groups[score]
        mass = sum(masses[cell] for cell in cells)
        fraction = min(1.0, max(0.0, remaining / mass))
        route[cells] = fraction
        remaining = max(0.0, remaining - mass * fraction)
    return route


def policy_metrics(routes, masses, probability, measured):
    weighted = np.asarray(routes) * masses
    calls = weighted.sum(axis=-1)
    missed = ((1 - np.asarray(routes)) * masses * (1 - probability)).sum(axis=-1)
    correct = weighted @ measured[:, 0]
    fee = weighted @ measured[:, 1]
    seconds = weighted @ measured[:, 2]
    return np.stack((calls, missed, correct, calls - correct, fee, seconds), axis=-1)


def summarize(configs, rows, sample):
    results = []
    for name, config in configs.items():
        sub = [r for r in rows if r["configuration"] == name]
        results.append(
            {
                "sample": sample,
                "configuration": name,
                "label": config["label"],
                "n": len(sub),
                "tasks": len({r["task_id"] for r in sub}),
                "correct": sum(r["answer_exact"] for r in sub),
                "accuracy": float(np.mean([r["answer_exact"] for r in sub])),
                "schema_valid": sum(r["schema_valid"] for r in sub),
                "source_ids_valid": sum(r["source_ids_valid"] for r in sub),
                "api_fee_estimate_usd": sum(r["api_fee_estimate_usd"] for r in sub),
                "api_fee_estimate_usd_per_1000_calls": float(
                    np.mean([r["api_fee_estimate_usd"] for r in sub]) * 1000
                ),
                "total_compute_cost_usd": None,
                "mean_latency_s": float(np.mean([r["latency_s"] for r in sub])),
                "median_latency_s": float(np.median([r["latency_s"] for r in sub])),
            }
        )
    return results


def paired_accuracy(configs, tasks, rows, sample):
    ids = sorted({r["task_id"] for r in rows})
    strata = defaultdict(list)
    for index, task_id in enumerate(ids):
        task = tasks[task_id]
        strata[task["family"], task["structural_level"]].append(index)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = np.concatenate(
        [
            rng.choice(cells, size=(BOOTSTRAP_DRAWS, len(cells)), replace=True)
            for cells in strata.values()
        ],
        axis=1,
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"], row["task_id"]].append(row["answer_exact"])
    values = {
        name: np.array([np.mean(grouped[name, task_id]) for task_id in ids]) for name in configs
    }
    results = []
    for first, second in combinations(configs, 2):
        delta = values[second] - values[first]
        lo, hi = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        results.append(
            {
                "sample": sample,
                "first": first,
                "second": second,
                "tasks": len(ids),
                "accuracy_delta": float(delta.mean()),
                "delta_lo": float(lo),
                "delta_hi": float(hi),
            }
        )
    return results


def replay(root, configs, tasks, rows, sample):
    population = sorted(read_csv(root / "data/population.csv"), key=lambda r: int(r["cell_id"]))
    masses = np.array([float(r["target_mass"]) for r in population])
    probability = np.array([float(r["probability"]) for r in population])
    predictions = defaultdict(lambda: np.full((200, 12), np.nan))
    for row in read_csv(root / "data/predictions.csv"):
        predictions[float(row["mu"]), row["method"]][int(row["seed"]), int(row["cell_id"])] = float(
            row["prediction"]
        )
    for mu in sorted({key[0] for key in predictions}):
        predictions[mu, "Oracle"] = np.tile(probability, (200, 1))
    measured = {}
    for name in configs:
        cells = defaultdict(list)
        for row in rows:
            if row["configuration"] == name:
                cells[tasks[row["task_id"]]["cell_id"]].append(
                    [row["answer_exact"], row["api_fee_estimate_usd"], row["latency_s"]]
                )
        measured[name] = np.array([np.mean(cells[cell], axis=0) for cell in range(12)])
    policies = []
    values = {}
    direct_error = 0.0
    for (mu, method), scores in sorted(predictions.items()):
        if not np.isfinite(scores).all():
            raise ValueError("Incomplete predictions")
        for rule, grid in [("threshold", THRESHOLDS), ("equal_call_budget", BUDGETS)]:
            for value in grid:
                routes = (
                    (scores < value).astype(float)
                    if rule == "threshold"
                    else np.array([budget_route(q, masses, value) for q in scores])
                )
                for name, means in measured.items():
                    observations = policy_metrics(routes, masses, probability, means)
                    if rule == "equal_call_budget" and not np.allclose(observations[:, 0], value):
                        raise ValueError("Call budget violated")
                    mean = observations.mean(axis=0)
                    values[name, mu, method, rule, value] = mean
                    policies.append(
                        {
                            "sample": sample,
                            "configuration": name,
                            "mu": mu,
                            "method": method,
                            "rule": rule,
                            "value": value,
                            **dict(zip(METRICS, map(float, mean))),
                            "correct_given_call": float(mean[2] / mean[0])
                            if mean[0] > 1e-12
                            else None,
                            "total_compute_cost_usd_per_case": None,
                        }
                    )
                    if rule == "equal_call_budget" and value == 0.5:
                        selected = [r for r in rows if r["configuration"] == name]
                        counts = defaultdict(int)
                        for row in selected:
                            counts[tasks[row["task_id"]]["cell_id"]] += 1
                        direct = np.zeros(3)
                        for row in selected:
                            cell = tasks[row["task_id"]]["cell_id"]
                            weight = masses[cell] * routes[:, cell].mean() / counts[cell]
                            direct += weight * np.array(
                                [row["answer_exact"], row["api_fee_estimate_usd"], row["latency_s"]]
                            )
                        direct_error = max(
                            direct_error, float(np.abs(direct - mean[[2, 4, 5]]).max())
                        )
    differences = []
    negative_error = 0.0
    for (name, mu, method, rule, value), means in values.items():
        delta = means - values[name, mu, "Raw", rule, value]
        differences.append(
            {
                "sample": sample,
                "configuration": name,
                "mu": mu,
                "method": method,
                "rule": rule,
                "value": value,
                **dict(zip(("delta_" + n for n in METRICS), map(float, delta))),
            }
        )
        if rule == "equal_call_budget" and method in ("Global", "Weighted global"):
            negative_error = max(negative_error, float(np.abs(delta).max()))
    if negative_error > 1e-12 or direct_error > 1e-10:
        raise ValueError("Policy replay control failed")
    return (
        policies,
        differences,
        {
            "negative_control_max_difference": negative_error,
            "direct_replay_max_difference": direct_error,
        },
    )


def analyze(root, output, configs, tasks, rows):
    output.mkdir(parents=True, exist_ok=False)
    summaries, pairs, policies, differences = [], [], [], []
    checks = {}
    for sample, subset in [
        ("full_288", rows),
        ("first_repeat_144", [r for r in rows if r["repeat"] == 1]),
    ]:
        summaries.extend(summarize(configs, subset, sample))
        pairs.extend(paired_accuracy(configs, tasks, subset, sample))
        policy, delta, control = replay(root, configs, tasks, subset, sample)
        policies.extend(policy)
        differences.extend(delta)
        checks[sample] = control
    for name, data in [
        ("model_summary", summaries),
        ("paired_accuracy", pairs),
        ("policy_summary", policies),
        ("policy_comparisons_vs_raw", differences),
    ]:
        write_csv(output / (name + ".csv"), data)
    failures = [
        {
            "configuration": r["configuration"],
            "task_id": r["task_id"],
            "repeat": r["repeat"],
            "gold_answer": tasks[r["task_id"]]["gold_answer"],
            "model_answer": (r["parsed_response"] or {}).get("final_answer"),
            "schema_valid": r["schema_valid"],
            "finish_reason": r["finish_reason"],
        }
        for r in rows
        if not r["answer_exact"]
    ]
    write_csv(output / "failures.csv", failures)
    return {
        "responses": len(rows),
        "configurations": len(configs),
        "unique_main_tasks": 144,
        "network_calls": 0,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "checks": checks,
    }
