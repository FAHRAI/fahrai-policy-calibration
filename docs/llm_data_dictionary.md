# LLM data dictionary

All paths below are relative to `data/llm/`.

| File | Contents |
|---|---|
| `tasks.json` | 156 synthetic exercises: 12 development and 144 main, gold answers, source rules, structural levels and synthetic cell metadata |
| `assignments.json` | Fixed order: 12 development assignments and two main repeats, 300 assignments total |
| `prompt.json` | Common instruction and JSON schema |
| `configurations.json` | Eight model configurations, historical token rates, local runtime, quantization and weight digest |
| `requests.jsonl` | 2,304 exact main request bodies; no credentials or HTTP headers |
| `responses.jsonl` | 2,304 final outputs with finish reason, recorded grades, token usage, estimated fees and latency |
| `reference/model_summary.csv` | Eight configurations for the complete main sample and first-repeat sensitivity sample |
| `reference/paired_accuracy.csv` | All 28 pairwise accuracy contrasts for each sample; delta is second minus first |
| `reference/policy_summary.csv` | Both samples, all configurations, logging settings, methods, thresholds and call budgets |
| `reference/policy_comparisons_vs_raw.csv` | Policy-metric differences relative to Raw at the same setting and budget or threshold |
| `reference/failures.csv` | Every main response failing the primary grading rule |
| `SHA256.json` | Checksums of the supplement, plus the unchanged population and prediction inputs |

`configuration` identifies the complete model/run setting, not a universally fixed model capability. `task_id` identifies an exercise; `assignment_id` appends `:r1` or `:r2` for its repetition. `phase` is `main` in every published response. The cell mapping and learner probabilities appear in the task file for analysis only; they are absent from model requests.

`response_text` is the final textual output, including empty or malformed output where received. Private reasoning channels, account details and provider request identifiers are omitted. `response_sha256` hashes its UTF-8 bytes. `request_sha256` hashes a JSON serialization with sorted keys, no ASCII escaping, and compact separators. It can be checked against `requests.jsonl`.

`parsed_response` is null if the response did not finish normally or its final text could not be parsed. `schema_valid`, `answer_exact` and `source_ids_valid` preserve the exact collection-time grading rule described in [the protocol](llm_experiment.md). `answer_exact` includes schema validity; it is not a number-only score.

`usage` records input, output, cached input, cache creation and available thinking-token counts. Null thinking usage means unavailable, not zero. `api_fee_estimate_usd` is computed with recorded historical rates and conservative cache accounting. `total_compute_cost_usd` is null throughout: full hardware, energy and service costs were not measured. A zero local API fee is not a zero total cost.

`local_runtime` exists only for local responses. Durations are seconds, memory allocations are bytes and throughput is generated tokens per second. Request latency includes prompt processing and any loading. Hardware and concurrency differ between configurations.

The reference analysis uses `full_288` and `first_repeat_144` to distinguish the two samples. Policy values are per target case. `correct_given_call` is null when the call rate is zero. CSV empty fields represent missing or undefined values, not zero.
